import torch
import torch.nn as nn
import torch.nn.functional as F
import metaspore as ms
import sys, json

from ...layers import MLPLayer

class MtlMMoEModel(torch.nn.Module):
    def __init__(self,
                 embedding_dim=8,
                 column_name_path=None,
                 combine_schema_path=None,
                 expert_numb=2,
                 task_numb=2,
                 expert_hidden_units=[512, 256],
                 expert_out_dim=16,
                 gate_hidden_units=[],
                 tower_hidden_units=[],
                 dnn_activations='ReLU',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=True,
                 net_dropout=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 use_gradnorm=False,  # Add flag
                 gradnorm_alpha=0.001, # Add alpha for gradnorm weight optimizer
                 use_uncertainty_weighting=False, #用于控制是否使用不确定性权重
                 gate_l2_reg_lambda=0.0, # 用于控制 Gate L2 正则化强度
                 expert_dissim_reg_lambda=0.0, # 用于控制 Expert Dissimilarity Loss 强度
                 expert_f_norm_reg_lambda=0.0, # 用于控制 F-norm 正则化强度
                 load_balancing_reg_lambda=0.0, # 用于控制 Load Balancing Loss 强度
                 temperature=1.0  # 温度参数，用于控制 Gate Softmax 的平滑程度
                 ):
        super().__init__()
        self.expert_numb = expert_numb
        self.task_numb = task_numb
        self.expert_out_dim = expert_out_dim
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.gate_l2_reg_lambda = gate_l2_reg_lambda
        self.expert_dissim_reg_lambda = expert_dissim_reg_lambda # 新增参数
        self.expert_f_norm_reg_lambda = expert_f_norm_reg_lambda # 新增参数
        self.load_balancing_reg_lambda = load_balancing_reg_lambda # 新增参数
        self.temperature = temperature # 新增温度参数
        self.use_gradnorm = use_gradnorm
        self.gradnorm_alpha = gradnorm_alpha # Store alpha for manager

        self.sparse = ms.EmbeddingSumConcat(embedding_dim, column_name_path, combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1 = ftrl_l1, l2 = ftrl_l2, alpha = ftrl_alpha, beta = ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.input_dim = int(self.sparse.feature_count * embedding_dim)

        self.experts = nn.ModuleList()
        for i in range(0, self.expert_numb):
            mlp = MLPLayer(input_dim = self.input_dim,
                            output_dim = self.expert_out_dim,
                            hidden_units = expert_hidden_units,
                            hidden_activations = dnn_activations,
                            final_activation = None,
                            dropout_rates = net_dropout,
                            input_norm=input_norm,
                            batch_norm = batch_norm,
                            use_bias = use_bias)
            self.experts.append(mlp)

        self.gates = nn.ModuleList()
        for i in range(0, self.task_numb):
            mlp = MLPLayer(input_dim = self.input_dim,
                            output_dim = self.expert_numb,
                            hidden_units = gate_hidden_units,
                            hidden_activations = dnn_activations,
                            final_activation = None,
                            dropout_rates = net_dropout,
                            input_norm=input_norm,
                            batch_norm = batch_norm,
                            use_bias = use_bias)
            self.gates.append(mlp)

        self.towers = nn.ModuleList()
        for i in range(0, self.task_numb):
            mlp = MLPLayer(input_dim = self.expert_out_dim,
                            output_dim = 1,
                            hidden_units = tower_hidden_units,
                            hidden_activations = dnn_activations,
                            final_activation = 'Sigmoid',
                            dropout_rates = net_dropout,
                            input_norm = input_norm,
                            batch_norm = batch_norm,
                            use_bias = use_bias)
            self.towers.append(mlp)

        initial_log_var_value = 0.0
        # Uncertainty Weighting相关参数
        if self.use_uncertainty_weighting:
            # 不确定性权重参数 - 作为可训练参数
            # 通过 register_parameter 注册的参数可以通过 model.log_vars 直接访问，并且会自动包含在 model.parameters() 中，参与梯度计算和参数更新。
            self.register_parameter('log_vars', nn.Parameter(torch.full((self.task_numb,), initial_log_var_value)))
        
        # 无条件注册缓冲区，确保 forward 中总能访问到
        self.register_buffer('total_gate_l2_norm', torch.tensor(0.0))
        # 新增：注册 Expert Dissimilarity Loss 缓冲区
        self.register_buffer('expert_dissim_loss', torch.tensor(0.0))
        # 新增：注册 F-norm 正则化缓冲区
        self.register_buffer('expert_f_norm_reg', torch.tensor(0.0))
        # 新增：注册 Load Balancing Loss 缓冲区
        self.register_buffer('load_balancing_loss', torch.tensor(0.0))

        # --- GradNorm Specific Initialization ---
        if self.use_gradnorm:
            self.task_weights = torch.nn.Parameter(torch.ones(self.task_numb), requires_grad=True)
        # ----------------------------------------

    def get_shared_params(self):
        """Returns parameters from layers shared across tasks (Experts and Gates)."""
        params = []
        params.extend(self.experts.parameters())
        params.extend(self.gates.parameters())
        return params
        
    def forward(self, x):
        x = self.sparse(x)
        expert_outputs = []
        for i in range(0, self.expert_numb):
            expert_out = self.experts[i](x)
            expert_outputs.append(expert_out)
        expert_cat = torch.cat(expert_outputs, dim=1)
        expert_cat = expert_cat.reshape(-1, self.expert_numb, self.expert_out_dim)

        predictions = []
        gate_raw_outputs = [] # 存储未经过 Softmax 的原始 Gate 输出
        gate_softmax_outputs = [] # 存储经过带温度Softmax的 Gate 输出，用于 Load Balancing Loss
        for i in range(0, self.task_numb):
            gate_out = self.gates[i](x)
            gate_raw_outputs.append(gate_out) # 收集原始输出
            
            # 使用温度参数的Softmax
            gate_out_softmax = F.softmax(gate_out / self.temperature, dim=1)
            gate_softmax_outputs.append(gate_out_softmax) # 收集 Softmax 输出
            
            gate_out_softmax = gate_out_softmax.reshape(-1, self.expert_numb, 1)
            tower_input = torch.mul(expert_cat, gate_out_softmax)
            tower_input = torch.sum(tower_input, 1)
            tower_out = self.towers[i](tower_input)
            predictions.append(tower_out)

        prediction = torch.cat(predictions, dim=1)

        # 计算所有 Gate 原始输出的 L2 范数之和
        if self.gate_l2_reg_lambda > 0: # 只有在需要时才计算，节省资源
            total_gate_l2_norm_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            for gate_raw_out in gate_raw_outputs:
                # gate_raw_out shape: (batch_size, expert_numb)
                # 计算每个样本的 L2 范数
                sample_l2_norms = torch.norm(gate_raw_out, p=2, dim=1) # shape: (batch_size,)
                total_gate_l2_norm_val = total_gate_l2_norm_val + torch.sum(sample_l2_norms)
            # Update the existing buffer
            self.total_gate_l2_norm = total_gate_l2_norm_val
        else:
            # Ensure buffer is zero if regularization is off
            self.total_gate_l2_norm = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # --- 新增：计算 Expert Dissimilarity Loss ---
        if self.expert_dissim_reg_lambda > 0 and self.expert_numb > 1:
            # 计算专家输出之间的余弦相似度
            # expert_outputs 是一个包含 self.expert_numb 个 tensor 的列表
            # 每个 tensor shape: (batch_size, expert_out_dim)
            
            # 将所有专家输出堆叠起来， shape: (batch_size, expert_numb, expert_out_dim)
            stacked_experts = torch.stack(expert_outputs, dim=1) # [B, E, D]
            
            dissim_loss_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
            # 遍历所有专家对 (i, j), i < j
            for i in range(self.expert_numb):
                for j in range(i + 1, self.expert_numb):
                    exp_i = stacked_experts[:, i, :] # [B, D]
                    exp_j = stacked_experts[:, j, :] # [B, D]
                    
                    # 计算批次中每个样本的余弦相似度, shape: (batch_size,)
                    cos_sim = F.cosine_similarity(exp_i, exp_j, dim=-1) 
                    
                    # 累加所有样本的相似度
                    dissim_loss_val = dissim_loss_val + torch.sum(cos_sim)
            
            # 更新缓冲区
            self.expert_dissim_loss = dissim_loss_val
        else:
            # 如果未启用或专家数不足，设置为零
            self.expert_dissim_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # --- 结束新增 ---

        # --- 新增：计算 F-norm 正则化 ---
        if self.expert_f_norm_reg_lambda > 0:
            # 根据 MLPLayer 实现，我们知道它的最后一层是一个 nn.Linear 层，
            # 其权重矩阵是输出层的权重。
            # 我们需要获取所有专家网络最后一层的权重矩阵，然后求和，再计算F范数的平方。
            
            # MLPLayer 的结构是 Sequential(*dense_layers)
            # 最后一层是 nn.Linear(hidden_units[-1], output_dim, bias=use_bias)
            # 即 self.experts[i].dnn[-1] 是最后一个Linear层
            # 获取其权重: self.experts[i].dnn[-1].weight
            
            total_W_sum = torch.zeros_like(self.experts[0].dnn[-1].weight, device=x.device, dtype=x.dtype)
            
            for expert_mlp in self.experts:
                # 获取专家MLP最后一层（Linear层）的权重
                last_linear_layer = expert_mlp.dnn[-1]  # 最后一个模块应该是 nn.Linear
                if isinstance(last_linear_layer, torch.nn.Linear):
                    # 累加权重矩阵
                    total_W_sum += last_linear_layer.weight
        
            # 计算Frobenius范数的平方
            f_norm_squared = torch.norm(total_W_sum, p='fro') ** 2
            
            # 更新缓冲区
            self.expert_f_norm_reg = f_norm_squared
        else:
            # 如果未启用，设置为零
            self.expert_f_norm_reg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # --- 结束新增 ---

        # --- 新增：计算 Load Balancing Loss (按MMoE论文标准实现) ---
        if self.load_balancing_reg_lambda > 0:
            # 对于每个任务，计算其Gate的负载均衡损失
            total_bal_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
            for gate_softmax_out in gate_softmax_outputs:
                # gate_softmax_out shape: (batch_size, expert_numb)
                
                # 1. 计算每个专家的平均选择概率 p_k^i = (1/N) * sum(P^i_{n,k})
                # 对批次维度求均值 -> shape: (expert_numb,)
                p_k_i = torch.mean(gate_softmax_out, dim=0)  # p_k^i
                
                # 2. 定义利用率 q_k^i = p_k^i (按照描述)
                q_k_i = p_k_i  # q_k^i = p_k^i
                
                # 3. 计算任务 i 的负载均衡损失 L_{balancing}^i = sum_k(p_k^i * q_k^i) = sum_k((p_k^i)^2)
                L_bal_i = torch.sum(p_k_i * q_k_i)  # = torch.sum(p_k_i ** 2)
                
                # 累加所有任务的负载均衡损失
                total_bal_loss = total_bal_loss + L_bal_i
            
            # 更新缓冲区
            self.load_balancing_loss = total_bal_loss
        else:
            # 如果未启用，设置为零
            self.load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # --- 结束新增 ---

        return prediction

    def get_task_weights(self):
        """获取当前任务权重（用于监控训练过程）"""
        if self.use_uncertainty_weighting and self.log_vars is not None:
            weights = []
            for i in range(self.task_numb):
                weight = torch.exp(-self.log_vars[i])
                weights.append(weight)
            return [w.item() for w in weights]
        else:
            return [1.0 for _ in range(self.task_numb)]

    def get_task_uncertainties(self):
        """获取当前任务不确定性（log_var值）"""
        if self.use_uncertainty_weighting and self.log_vars is not None:
            uncertainties = []
            for i in range(self.task_numb):
                uncertainties.append(self.log_vars[i])
            return [u.item() for u in uncertainties]
        else:
            return [0.0 for _ in range(self.task_numb)]

    def predict(self, yhat, minibatch = None):
        #返回第一个任务的预测结果
        return yhat[:, 0]

    def compute_uncertainty_loss(self, y_pred, y_true, minibatch, task_to_id_map=None):
        """
        支持不确定性权重的多标签损失函数
        """
        if task_to_id_map is None:
            raise ValueError("task_to_id_map must be provided.")

        y_true_list = minibatch['mul_labels']
        batch_size = y_pred.size(0)
        num_tasks = len(task_to_id_map)
        device = y_pred.device

        target_tensor = torch.full((batch_size, num_tasks), -1.0, dtype=torch.float32, device=device)
        for batch_idx, label_map in enumerate(y_true_list):
            if isinstance(label_map, str):
                try:
                    label_dict = json.loads(label_map)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(label_map, dict):
                label_dict = label_map
            else:
                continue

            if isinstance(label_dict, dict):
                for task_name, label_val in label_dict.items():
                    if task_name in task_to_id_map:
                        task_id = task_to_id_map[task_name]
                        target_tensor[batch_idx, task_id] = float(label_val)

        bce_loss = F.binary_cross_entropy(y_pred, target_tensor, reduction='none')
        mask = (target_tensor != -1.0)

        if mask.sum() == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            per_task_losses_for_monitoring = torch.zeros(num_tasks, device=device)
            return total_loss, per_task_losses_for_monitoring

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        individual_losses_with_grad = [] # Store raw individual losses WITH gradients

        if self.use_uncertainty_weighting and self.log_vars is not None:
            for task_id in range(num_tasks):
                task_mask = mask[:, task_id]
                num_valid = task_mask.sum()

                if num_valid > 0:
                    avg_bce = bce_loss[:, task_id][task_mask].mean()
                    individual_losses_with_grad.append(avg_bce) # Keep grad

                    if task_id < len(self.log_vars):
                        log_var = self.log_vars[task_id]
                        precision = torch.exp(-log_var)
                        task_loss = precision * avg_bce + log_var
                    else:
                        task_loss = avg_bce

                    total_loss = total_loss + task_loss
                else:
                    individual_losses_with_grad.append(torch.tensor(0.0, device=device, requires_grad=True)) # Keep grad
                    # No contribution to total_loss if no valid samples

        else:
            # If not using uncertainty weighting, calculate simple weighted BCE based on GradNorm weights
            if self.use_gradnorm:
                # For GradNorm, we need the *individual* BCE losses WITH GRADIENTS
                # These will be used by GradNorm manager to calculate grad norms and update weights
                for task_id in range(num_tasks):
                    task_mask = mask[:, task_id]
                    if task_mask.sum() > 0:
                        avg_bce = bce_loss[:, task_id][task_mask].mean()
                        individual_losses_with_grad.append(avg_bce) # Keep grad
                    else:
                        # If no valid samples for a task, add a dummy zero loss
                        individual_losses_with_grad.append(torch.tensor(0.0, device=device, requires_grad=True)) # Keep grad

                # Now apply GradNorm weights to these individual losses for the main model parameter update
                weighted_losses = [self.task_weights[t] * individual_losses_with_grad[t] for t in range(len(individual_losses_with_grad))]
                total_loss = sum(weighted_losses) # This loss is used for main model update
                # The detached version can be returned for monitoring if needed
            else:
                # Standard weighted loss (could be uniform weights)
                for task_id in range(num_tasks):
                    task_mask = mask[:, task_id]
                    if task_mask.sum() > 0:
                        avg_bce = bce_loss[:, task_id][task_mask].mean()
                        total_loss = total_loss + avg_bce # Uniform weighting if no strategy
                        individual_losses_with_grad.append(avg_bce) # Keep grad for consistency
                    else:
                        individual_losses_with_grad.append(torch.tensor(0.0, device=device, requires_grad=True)) # Keep grad

        # For GradNorm, return the losses WITH gradients
        # For monitoring, you can call .detach() outside this function
        per_task_losses_for_gradnorm = torch.stack(individual_losses_with_grad)
        # Optionally, also return a detached version for logging/monitoring if needed elsewhere
        # per_task_losses_for_monitoring = per_task_losses_for_gradnorm.detach()

        return total_loss, per_task_losses_for_gradnorm # Return the one with grad

    def compute_loss(self, predictions, labels, minibatch, task_to_id_map = None):
        # 1. Compute the primary multi-task prediction loss
        # This includes handling uncertainty weighting internally.
        primary_prediction_loss, per_task_original_losses = self.compute_uncertainty_loss(
            predictions, labels.float(), minibatch, task_to_id_map
        )

        # 2. Compute auxiliary regularization losses (same as before)
        aux_reg_loss = (
            self.gate_l2_reg_lambda * self.total_gate_l2_norm +
            self.expert_dissim_reg_lambda * self.expert_dissim_loss +
            self.expert_f_norm_reg_lambda * self.expert_f_norm_reg +
            self.load_balancing_reg_lambda * self.load_balancing_loss
        )

        # 3. Total loss for the main model update
        # If using GradNorm, primary_prediction_loss already contains the weighted sum.
        # If using Uncertainty Weighting, primary_prediction_loss already contains the weighted sum.
        # If using neither, primary_prediction_loss is just the sum of unweighted BCEs.
        total_loss = primary_prediction_loss + aux_reg_loss

        # Return total loss for main model update and the original prediction loss for monitoring
        return total_loss, primary_prediction_loss