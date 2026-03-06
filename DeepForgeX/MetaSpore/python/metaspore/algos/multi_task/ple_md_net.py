import torch
import torch.nn as nn
import metaspore as ms

from ..layers import MLPLayer

class ExpertNetwork(torch.nn.Module):
    """专家网络"""
    def __init__(self, input_dim, expert_hidden_units, output_dim=None, activation='ReLU', dropout_rate=None, batch_norm=False):
        super().__init__()
        self.mlp = MLPLayer(
            input_dim=input_dim,
            output_dim=output_dim or expert_hidden_units[-1],
            hidden_units=expert_hidden_units[:-1] if output_dim is None else expert_hidden_units,
            hidden_activations=activation,
            final_activation=activation if output_dim is None else None,
            dropout_rates=dropout_rate,
            batch_norm=batch_norm
        )
    
    def forward(self, x):
        return self.mlp(x)

class PLEGate(torch.nn.Module):
    """PLE门控机制"""
    def __init__(self, input_dim, num_experts, gate_hidden_units=None, activation='ReLU', dropout_rate=None, batch_norm=False):
        super().__init__()
        self.num_experts = num_experts
        gate_units = gate_hidden_units or [64]
        
        self.gate_mlp = MLPLayer(
            input_dim=input_dim,
            output_dim=num_experts,
            hidden_units=gate_units,
            hidden_activations=activation,
            final_activation=None,
            dropout_rates=dropout_rate,
            batch_norm=batch_norm
        )
    
    def forward(self, x):
        gate_weights = self.gate_mlp(x)
        gate_weights = torch.softmax(gate_weights, dim=-1)
        return gate_weights

class TaskSpecificExpert(torch.nn.Module):
    """任务特定专家网络"""
    def __init__(self, input_dim, expert_hidden_units, activation='ReLU', dropout_rate=None, batch_norm=False):
        super().__init__()
        self.expert = ExpertNetwork(input_dim, expert_hidden_units, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm)
    
    def forward(self, x):
        return self.expert(x)

class SharedExpert(torch.nn.Module):
    """共享专家网络"""
    def __init__(self, input_dim, expert_hidden_units, activation='ReLU', dropout_rate=None, batch_norm=False):
        super().__init__()
        self.expert = ExpertNetwork(input_dim, expert_hidden_units, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm)
    
    def forward(self, x):
        return self.expert(x)

class DomainSpecificExpert(torch.nn.Module):
    """领域特定专家网络"""
    def __init__(self, input_dim, expert_hidden_units, activation='ReLU', dropout_rate=None, batch_norm=False):
        super().__init__()
        self.expert = ExpertNetwork(input_dim, expert_hidden_units, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm)
    
    def forward(self, x):
        return self.expert(x)

class SinglePLEMDLayer(torch.nn.Module):
    """单层PLE-MD层 - 多场景多任务网络层"""
    def __init__(self, 
                 input_dim,
                 domain_count,
                 task_count,
                 shared_expert_num=4,
                 domain_expert_num=2,
                 task_expert_num=2,
                 expert_hidden_units=[256, 128],
                 gate_hidden_units=[64],
                 activation='ReLU',
                 dropout_rate=None,
                 batch_norm=False):
        super().__init__()
        
        self.domain_count = domain_count
        self.task_count = task_count
        self.shared_expert_num = shared_expert_num
        self.domain_expert_num = domain_expert_num
        self.task_expert_num = task_expert_num
        
        # 共享专家网络
        self.shared_experts = torch.nn.ModuleList([
            SharedExpert(input_dim, expert_hidden_units, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm)
            for _ in range(shared_expert_num)
        ])
        
        # 领域特定专家网络 (每个领域都有自己的专家)
        self.domain_experts = torch.nn.ModuleDict()
        for domain_id in range(domain_count):
            experts = torch.nn.ModuleList([
                DomainSpecificExpert(input_dim, expert_hidden_units, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm)
                for _ in range(domain_expert_num)
            ])
            self.domain_experts[str(domain_id)] = experts
        
        # 任务特定专家网络 (每个任务都有自己的专家)
        self.task_experts = torch.nn.ModuleDict()
        for task_id in range(task_count):
            experts = torch.nn.ModuleList([
                TaskSpecificExpert(input_dim, expert_hidden_units, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm)
                for _ in range(task_expert_num)
            ])
            self.task_experts[str(task_id)] = experts
        
        # 门控网络 - 共享专家门控
        self.shared_gate = PLEGate(input_dim, shared_expert_num, gate_hidden_units, activation, dropout_rate, batch_norm)
        
        # 领域特定门控网络
        self.domain_gates = torch.nn.ModuleDict()
        for domain_id in range(domain_count):
            total_experts = shared_expert_num + domain_expert_num
            gate = PLEGate(input_dim, total_experts, gate_hidden_units, activation, dropout_rate, batch_norm)
            self.domain_gates[str(domain_id)] = gate
        
        # 任务特定门控网络
        self.task_gates = torch.nn.ModuleDict()
        for task_id in range(task_count):
            # 每个任务看到：共享专家 + 所有领域的专家 + 自身任务专家
            total_experts = shared_expert_num + domain_expert_num * domain_count + task_expert_num
            gate = PLEGate(input_dim, total_experts, gate_hidden_units, activation, dropout_rate, batch_norm)
            self.task_gates[str(task_id)] = gate
    
    def forward(self, x, domain_ids):
        batch_size = x.size(0)
        
        # 获取共享专家输出 - 对整个批次一次性计算
        shared_outputs = []
        for expert in self.shared_experts:
            shared_outputs.append(expert(x))
        shared_outputs = torch.stack(shared_outputs, dim=1)  # [batch_size, shared_expert_num, expert_output_dim]
        
        # 获取所有领域特定专家输出 - 对整个批次一次性计算
        all_domain_experts_outputs = []
        for domain_idx in range(self.domain_count):
            domain_outputs = []
            domain_experts_for_current_domain = self.domain_experts[str(domain_idx)]
            for expert in domain_experts_for_current_domain:
                domain_outputs.append(expert(x))
            domain_outputs = torch.stack(domain_outputs, dim=1)  # [batch_size, domain_expert_num, expert_output_dim]
            all_domain_experts_outputs.append(domain_outputs)
        
        all_domain_specific_outputs = torch.cat(all_domain_experts_outputs, dim=1)  # [batch_size, domain_count*domain_expert_num, expert_output_dim]
        
        # 计算任务特定专家输出 - 对整个批次一次性计算
        all_task_specific_outputs = []
        for task_idx in range(self.task_count):
            task_specific_outputs = []
            task_experts_for_current_task = self.task_experts[str(task_idx)]
            for expert in task_experts_for_current_task:
                task_specific_outputs.append(expert(x))
            task_specific_outputs = torch.stack(task_specific_outputs, dim=1)  # [batch_size, task_expert_num, expert_output_dim]
            all_task_specific_outputs.append(task_specific_outputs)
        
        # 为每个样本分别处理领域和任务输出
        task_outputs_list = [[] for _ in range(self.task_count)]
        
        for sample_idx in range(batch_size):
            current_domain_id = domain_ids[sample_idx].item()
            
            # 获取当前样本的领域特定输出
            current_sample_x = x[sample_idx:sample_idx+1]  # [1, input_dim]
            
            # 当前领域的专家输出
            current_domain_outputs = []
            domain_experts_for_current = self.domain_experts[str(current_domain_id)]
            for expert in domain_experts_for_current:
                current_domain_outputs.append(expert(current_sample_x))
            current_domain_outputs = torch.stack(current_domain_outputs, dim=1)  # [1, domain_expert_num, expert_output_dim]
            
            # 当前领域的门控加权
            domain_gate_input = current_sample_x
            domain_gate_weights = self.domain_gates[str(current_domain_id)](domain_gate_input)  # [1, total_domain_experts]
            all_current_domain_experts = torch.cat([
                shared_outputs[sample_idx:sample_idx+1],  # [1, shared_expert_num, expert_output_dim]
                current_domain_outputs  # [1, domain_expert_num, expert_output_dim]
            ], dim=1)  # [1, shared+domain, expert_dim]
            domain_weighted_output = torch.sum(all_current_domain_experts * domain_gate_weights.unsqueeze(-1), dim=1)  # [1, expert_dim]
            
            # 为当前样本计算每个任务的输出
            for task_idx in range(self.task_count):
                # 获取当前任务的特定专家输出
                current_task_specific_outputs = all_task_specific_outputs[task_idx][sample_idx:sample_idx+1]  # [1, task_expert_num, expert_output_dim]
                
                # 构建当前任务的所有专家
                all_current_task_experts = torch.cat([
                    shared_outputs[sample_idx:sample_idx+1],  # [1, shared_expert_num, expert_output_dim]
                    all_domain_specific_outputs[sample_idx:sample_idx+1],  # [1, domain_count*domain_expert_num, expert_output_dim]
                    current_task_specific_outputs  # [1, task_expert_num, expert_output_dim]
                ], dim=1)  # [1, shared+all_domain+task, expert_dim]
                
                # 当前任务的门控权重
                task_gate_input = current_sample_x
                task_gate_weights = self.task_gates[str(task_idx)](task_gate_input)  # [1, total_task_experts]
                
                # 当前任务的加权输出
                current_task_output = torch.sum(all_current_task_experts * task_gate_weights.unsqueeze(-1), dim=1)  # [1, expert_dim]
                task_outputs_list[task_idx].append(current_task_output)
        
        # 合并所有样本的任务输出
        final_task_outputs = []
        for task_idx in range(self.task_count):
            task_output_batch = torch.cat(task_outputs_list[task_idx], dim=0)  # [batch_size, expert_dim]
            final_task_outputs.append(task_output_batch)
        
        return final_task_outputs

class MultiLayerPLEMD(torch.nn.Module):
    """多层PLE-MD网络"""
    def __init__(self,
                 domain_count=3,                    # 领域数量
                 task_count=2,                      # 任务数量
                 embedding_dim=16,                  # 嵌入维度
                 num_layers=2,                      # PLE-MD层数
                 shared_expert_num=4,               # 共享专家数量
                 domain_expert_num=2,               # 每个领域的专家数量
                 task_expert_num=2,                 # 每个任务的专家数量
                 expert_hidden_units=[256, 128],          # 专家网络隐藏单元
                 gate_hidden_units=[64],           # 门控网络隐藏单元
                 task_towers_hidden_units=[64],    # 任务塔隐藏单元
                 column_name_path=None,            # 特征列名路径
                 combine_schema_path=None,         # 特征组合schema路径
                 sparse_init_var=1e-2,             # 稀疏初始化方差
                 activation='ReLU',                # 激活函数
                 batch_norm=False,                 # 是否使用批归一化
                 net_dropout=None,                 # 网络dropout率
                 ftrl_l1=1.0,                     # FTRL L1正则化
                 ftrl_l2=120.0,                   # FTRL L2正则化
                 ftrl_alpha=0.5,                  # FTRL alpha参数
                 ftrl_beta=1.0,
                 use_uncertainty_weighting=False):      
        super().__init__()
        
        self.domain_count = domain_count
        self.task_count = task_count
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # 共享嵌入层
        self.shared_embedding = ms.EmbeddingSumConcat(
            embedding_dim, 
            column_name_path, 
            combine_schema_path
        )
        self.shared_embedding.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, 
            l2=ftrl_l2, 
            alpha=ftrl_alpha, 
            beta=ftrl_beta
        )
        self.shared_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # 输入维度计算
        input_dim = int(self.shared_embedding.feature_count * embedding_dim)
        
        # 共享嵌入归一化
        self.shared_bn = ms.nn.Normalization(input_dim)
        
        # 创建多层PLE-MD
        self.ple_md_layers = torch.nn.ModuleList()
        
        # 第一层的输入维度是嵌入层的输出
        layer_input_dim = input_dim
        for i in range(num_layers):
            ple_layer = SinglePLEMDLayer(
                input_dim=layer_input_dim,
                domain_count=domain_count,
                task_count=task_count,
                shared_expert_num=shared_expert_num,
                domain_expert_num=domain_expert_num,
                task_expert_num=task_expert_num,
                expert_hidden_units=expert_hidden_units,
                gate_hidden_units=gate_hidden_units,
                activation=activation,
                dropout_rate=net_dropout,
                batch_norm=batch_norm
            )
            self.ple_md_layers.append(ple_layer)
            
            # 后续层的输入维度是专家网络的输出维度
            layer_input_dim = expert_hidden_units[-1]  # 专家网络的最后输出维度
        
        # 任务特定塔 - 接收最后一层PLE-MD的输出
        self.task_towers = torch.nn.ModuleList()
        for _ in range(task_count):
            tower = MLPLayer(
                input_dim=expert_hidden_units[-1],  # 最后一层PLE-MD的输出维度
                output_dim=1,
                hidden_units=task_towers_hidden_units,
                hidden_activations=activation,
                final_activation=None,  # 最后会应用sigmoid
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            self.task_towers.append(tower)
        
        # 最终激活函数
        self.final_activation = torch.nn.Sigmoid()
        self.use_uncertainty_weighting = use_uncertainty_weighting
        initial_log_var_value = 0.0
        # Added: Uncertainty Weighting相关参数
        if self.use_uncertainty_weighting:
            # 不确定性权重参数 - 作为可训练参数
            # 通过 register_parameter 注册的参数可以通过 model.log_vars 直接访问，并且会自动包含在 model.parameters() 中，参与梯度计算和参数更新。
            self.register_parameter('log_vars', nn.Parameter(torch.full((self.task_count,), initial_log_var_value)))
    
    def forward(self, x):
        # 获取当前批次的domain_id
        domain_ids = self._current_domain_ids
        
        # 共享嵌入
        shared_emb = self.shared_embedding(x)
        shared_emb_bn = self.shared_bn(shared_emb)
        
        # 多层PLE-MD处理
        current_input = shared_emb_bn
        for layer_idx, ple_layer in enumerate(self.ple_md_layers):
            # 每层PLE-MD都输出所有任务的中间表示
            task_outputs = ple_layer(current_input, domain_ids)
            
            # 对于多层，我们取第一个任务的输出作为下一层的输入（或者可以选择平均/拼接等方式）
            # 这里我们取所有任务输出的平均值作为下一层的输入
            next_input = torch.stack(task_outputs, dim=0).mean(dim=0)
            current_input = next_input
        
        # 使用最终层的输出进行任务预测
        task_predictions = []
        for i in range(self.task_count):
            task_output = self.task_towers[i](current_input)  # 使用最终的表示
            task_prediction = self.final_activation(task_output)
            task_predictions.append(task_prediction)
        
        # 合并所有任务预测结果
        prediction = torch.cat(task_predictions, dim=1)  # [batch_size, task_count]
        return prediction
    
    def get_task_weights(self):
        """获取当前任务权重（用于监控训练过程）"""
        if self.use_uncertainty_weighting and self.log_vars is not None:
            weights = []
            for i in range(self.task_count):
                weight = torch.exp(-self.log_vars[i])
                weights.append(weight)
            return [w.item() for w in weights]
        else:
            return [1.0 for _ in range(self.task_count)]

    def get_task_uncertainties(self):
        """获取当前任务不确定性（log_var值）"""
        if self.use_uncertainty_weighting and self.log_vars is not None:
            uncertainties = []
            for i in range(self.task_count):
                uncertainties.append(self.log_vars[i])
            return [u.item() for u in uncertainties]
        else:
            return [0.0 for _ in range(self.task_count)]

    def do_extra_work(self, minibatch):
        # 获取domain_id
        if isinstance(minibatch, dict):
            domain_id_col = minibatch.get('domain_id')
        elif hasattr(minibatch, 'domain_id'):  # 如 Spark Row 或 namedtuple
            domain_id_col = minibatch.domain_id
        else:
            # 如果是 pandas DataFrame
            domain_id_col = minibatch['domain_id'].values if 'domain_id' in minibatch.columns else None
        
        # 处理domain_id
        if domain_id_col is not None:
            if not isinstance(domain_id_col, torch.Tensor):
                domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
            else:
                domain_id_tensor = domain_id_col.long()
        else:
            batch_size = len(minibatch) if hasattr(minibatch, '__len__') else 1
            domain_id_tensor = torch.zeros(batch_size, dtype=torch.long)
        
        self._current_domain_ids = domain_id_tensor  # 保存到模型内部

    def predict(self, yhat, minibatch = None):
        #返回第一个任务的预测结果
        return yhat[:, 0]