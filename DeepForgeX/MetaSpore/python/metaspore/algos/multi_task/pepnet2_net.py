import torch
import torch.nn as nn
import metaspore as ms

class GateNULayer(torch.nn.Module):
    """GateNU层，带缩放因子的门控机制"""
    def __init__(self, input_dim, scale_factor=1.0):
        super(GateNULayer, self).__init__()
        self.scale_factor = scale_factor
        self.gate = torch.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        gate_logits = self.gate(x)
        scaled_logits = gate_logits * self.scale_factor
        gate_weights = torch.sigmoid(scaled_logits)
        return x * gate_weights


class EPNetLayer(torch.nn.Module):
    """嵌入个性化网络层 (Embedding Personalized Network)"""
    def __init__(self, input_dim, domain_count, embedding_dim=16):
        super(EPNetLayer, self).__init__()
        self.domain_count = domain_count
        # Domain-specific embeddings to generate gating weights
        self.domain_embeddings = torch.nn.Embedding(domain_count, embedding_dim)
        self.domain_to_gate = torch.nn.Linear(embedding_dim, input_dim)
        self.gate_nu = GateNULayer(input_dim, scale_factor=1.0)

    def forward(self, general_input, domain_ids):
        """
        通过domain-specific信息对general input进行个性化调整
        Args:
            general_input: [batch_size, input_dim] - 来自Shared Bottom的通用表示
            domain_ids: [batch_size] - 每个样本的domain id
        Returns:
            [batch_size, input_dim] - 个性化后的表示
        """
        # Get domain-specific embeddings
        domain_embeds = self.domain_embeddings(domain_ids)  # [B, E]

        # Transform domain embeddings to gate weights
        gate_weights_raw = self.domain_to_gate(domain_embeds)  # [B, D_SB]
        
        # Apply GateNU for final gating weights
        gate_weights = self.gate_nu(gate_weights_raw)  # [B, D_SB]

        # Apply gating to general input (from Shared Bottom)
        personalized_output = general_input * gate_weights  # [B, D_SB]
        return personalized_output


class PPNetLayer(torch.nn.Module):
    """参数个性化网络层 (Parameter Personalized Network) - Modified for All Tasks"""
    def __init__(self, prior_input_dim, general_input_dim, task_count, tower_hidden_units, scale_factor=1.0):
        """
        Args:
            prior_input_dim: 先验特征embedding的维度
            general_input_dim: 来自Shared Bottom的通用表示维度
            task_count: 任务数量
            tower_hidden_units: 任务塔的隐藏层维度列表，如[128, 64]
            scale_factor: GateNU缩放因子
        """
        super(PPNetLayer, self).__init__()
        self.task_count = task_count
        self.tower_hidden_units = tower_hidden_units

        # Combined feature dimension for generating gates (prior + shared_bottom_output)
        combined_input_dim = prior_input_dim + general_input_dim

        # --- Task Towers ---
        # Create all task towers using ModuleList for easier management
        self.task_towers = torch.nn.ModuleList()
        layer_dims = [general_input_dim] + list(tower_hidden_units) + [1] # Add output dim

        for _ in range(task_count):
            layers = []
            for i in range(len(layer_dims) - 1):
                layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
                if i < len(layer_dims) - 2:  # No ReLU after the last linear layer
                    layers.append(torch.nn.ReLU())
            self.task_towers.append(torch.nn.Sequential(*layers))

        # --- Parameter Personalization Networks (PPNs) ---
        # One PPN per task tower layer (excluding output layer)
        self.ppn_networks = torch.nn.ModuleList() # Outer list for tasks

        for task_id in range(task_count):
            task_ppns = torch.nn.ModuleList() # Inner list for layers of this task
            for i in range(len(layer_dims) - 2): # Exclude input and output layers for PPNs
                ppn = torch.nn.Sequential(
                    torch.nn.Linear(combined_input_dim, layer_dims[i+1]),
                    torch.nn.ReLU(),
                    GateNULayer(layer_dims[i+1], scale_factor)
                )
                task_ppns.append(ppn)
            self.ppn_networks.append(task_ppns)

    def forward(self, prior_emb, general_input, personalized_input):
        """
        为每个样本计算所有任务的预测logits
        Args:
            prior_emb: [batch_size, prior_input_dim] - 先验特征的embedding
            general_input: [batch_size, general_input_dim] - 来自Shared Bottom的通用表示
            personalized_input: [batch_size, general_input_dim] - 个性化embedding (由EPNet输出)
        Returns:
            torch.Tensor: [batch_size, task_count] - 每个样本在所有任务下的预测logits
        """
        batch_size = general_input.size(0)
        # Use the shared bottom output (general_input) and prior_emb for PPN
        combined_features = torch.cat([prior_emb, general_input], dim=1)  # [B, PE + D_SB]

        # Prepare lists to store outputs for each task
        all_task_outputs = []

        # Iterate over each task tower
        for task_id in range(self.task_count):
            task_tower = self.task_towers[task_id]
            task_ppns = self.ppn_networks[task_id]

            # Process the entire batch through this specific task tower
            # Start with the personalized embedding for this batch
            x = personalized_input # [B, D_SB]

            # Iterate through the layers of the current task tower
            layer_idx_in_tower = 0
            ppn_layer_idx = 0
            for layer in task_tower:
                x = layer(x) # Apply the layer's transformation
                # Check if this is a Linear layer that needs PPN gating
                # and if we have a corresponding PPN
                if isinstance(layer, torch.nn.Linear) and \
                   layer_idx_in_tower // 2 < len(task_ppns): # 2 layers per block (Linear + ReLU)
                    # Generate gate weights for this layer using combined features (from Shared Bottom)
                    gate_weights = task_ppns[ppn_layer_idx](combined_features) # [B, Out_Dim_of_Current_Layer]
                    # Apply gating
                    x = x * gate_weights
                    ppn_layer_idx += 1
                layer_idx_in_tower += 1
            
            # x now contains the final output (logits) for this task, shape [B, 1]
            all_task_outputs.append(x) # Append [B, 1]

        # Concatenate all task outputs along the task dimension
        # Resulting shape: [B, task_count, 1]
        concatenated_logits = torch.cat(all_task_outputs, dim=1) # [B, task_count]
        return concatenated_logits.squeeze(-1) # Remove the last dimension to get [B, task_count]


class PEPNet2(torch.nn.Module):
    """参数和嵌入个性化网络 (Parameter and Embedding Personalized Network) - With Shared Bottom MLP"""
    def __init__(self,
                 domain_count=3,
                 task_count=3,
                 embedding_dim=16,
                 base_dnn_hidden_units=[512, 256], # NEW: Define the Shared Bottom MLP layers
                 task_towers_hidden_units=[128, 64],
                 # Shared embedding params
                 column_name_path=None,
                 combine_schema_path=None,
                 # Prior embedding params
                 prior_column_name_path=None,
                 prior_combine_schema_path=None,
                 # Other params
                 sparse_init_var=1e-2,
                 scale_factor=1.0,
                 batch_norm=False,
                 net_dropout=None, # e.g., 0.1
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 use_uncertainty_weighting=False):
        super(PEPNet2, self).__init__()

        self.domain_count = domain_count
        self.task_count = task_count
        self.embedding_dim = embedding_dim
        self.scale_factor = scale_factor
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # --- Shared Embedding (General Features) ---
        self.shared_embedding = ms.EmbeddingSumConcat(
            embedding_dim,
            column_name_path,
            combine_schema_path)
        self.shared_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.shared_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        # --- Prior Embedding (User/Item/Author-side Features) ---
        self.prior_embedding = ms.EmbeddingSumConcat(
            embedding_dim,
            prior_column_name_path,
            prior_combine_schema_path)
        self.prior_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.prior_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        # --- Calculate dimensions ---
        shared_input_dim = self.shared_embedding.feature_count * embedding_dim
        prior_input_dim = self.prior_embedding.feature_count * embedding_dim
        # NEW: Calculate the output dimension of the Shared Bottom MLP
        shared_bottom_output_dim = base_dnn_hidden_units[-1] if base_dnn_hidden_units else shared_input_dim


        # --- Normalization for Shared Embedding ---
        self.bn_shared = ms.nn.Normalization(shared_input_dim)

        # --- Shared Bottom MLP ---
        # NEW: Create the shared bottom layers
        self.shared_bottom_layers = torch.nn.Sequential()
        layer_dims = [shared_input_dim] + list(base_dnn_hidden_units)
        for i in range(len(layer_dims) - 1):
            self.shared_bottom_layers.add_module(f'linear_{i}', torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
            if batch_norm:
                self.shared_bottom_layers.add_module(f'bn_{i}', ms.nn.Normalization(layer_dims[i+1]))
            self.shared_bottom_layers.add_module(f'relu_{i}', torch.nn.ReLU())
            if net_dropout:
                self.shared_bottom_layers.add_module(f'dropout_{i}', torch.nn.Dropout(net_dropout))

        # --- EPNet ---
        # NEW: EPNet now operates on the output of the Shared Bottom
        self.ep_net = EPNetLayer(input_dim=shared_bottom_output_dim, # Use shared_bottom_output_dim
                                 domain_count=domain_count,
                                 embedding_dim=embedding_dim)

        # --- PPNet ---
        # NEW: PPNet receives the output of the Shared Bottom as general_input
        self.pp_net = PPNetLayer(prior_input_dim=prior_input_dim,
                                 general_input_dim=shared_bottom_output_dim, # Use shared_bottom_output_dim
                                 task_count=task_count,
                                 tower_hidden_units=task_towers_hidden_units,
                                 scale_factor=scale_factor)

        # --- Final Activation ---
        self.final_activation = torch.nn.Sigmoid()

        initial_log_var_value = 0.0
        # Added: Uncertainty Weighting相关参数
        if self.use_uncertainty_weighting:
            # 不确定性权重参数 - 作为可训练参数
            # 通过 register_parameter 注册的参数可以通过 model.log_vars 直接访问，并且会自动包含在 model.parameters() 中，参与梯度计算和参数更新。
            self.register_parameter('log_vars', torch.nn.Parameter(torch.full((self.task_count,), initial_log_var_value)))

    def forward(self, x):
        """
        Model forward pass. Computes predictions for all tasks for each sample.
        Assumes `do_extra_work` has been called to set `self._current_*` attributes.
        """
        # Retrieve pre-processed data (set by do_extra_work)
        domain_ids = self._current_domain_ids  # [B]
        # x is the input for shared embedding [B, N]

        # --- 1. Shared Embedding & BN ---
        shared_emb = self.shared_embedding(x)  # [B, D_shared]
        shared_emb_bn = self.bn_shared(shared_emb)    # [B, D_shared]

        # --- 2. Shared Bottom MLP ---
        # NEW: Pass the normalized shared embedding through the Shared Bottom MLP
        shared_bottom_output = self.shared_bottom_layers(shared_emb_bn) # [B, D_SB]

        # --- 3. Prior Embedding ---
        prior_emb = self.prior_embedding(x) # [B, D_prior]

        # --- 4. EPNet: Embedding Personalization ---
        # NEW: EPNet now operates on the output of the Shared Bottom
        personalized_emb = self.ep_net(shared_bottom_output, domain_ids)  # [B, D_SB]

        # --- 5. PPNet: Parameter Personalization - For ALL Tasks ---
        # NEW: PPNet receives the output of the Shared Bottom as general_input
        logits = self.pp_net(prior_emb, shared_bottom_output, personalized_emb)  # [B, task_num]

        # --- 6. Final Activation ---
        predictions = self.final_activation(logits)  # [B, task_num]

        return predictions # Return the full [B, task_num] tensor

    def do_extra_work(self, minibatch):
        """
        Pre-processes the minibatch to extract domain_id only.
        Task IDs are no longer needed.
        """
        # --- Extract domain_id ---
        if hasattr(minibatch, 'domain_id'):
            domain_id_data = minibatch.domain_id
        elif isinstance(minibatch, dict) and 'domain_id' in minibatch:
            domain_id_data = minibatch['domain_id']
        else:
            domain_id_data = minibatch['domain_id'].values

        if not isinstance(domain_id_data, torch.Tensor):
            self._current_domain_ids = torch.tensor(domain_id_data, dtype=torch.long)
        else:
            self._current_domain_ids = domain_id_data.long()

        # No need to extract task_ids anymore
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

    def predict(self, yhat, minibatch=None):
        # 根据任务数量返回适当的预测结果
        if self.task_count == 1:
            # 单任务情况下，yhat已经是[B]或[B, 1]，直接返回第一个任务的结果
            if yhat.dim() == 2 and yhat.size(1) == 1:
                return yhat.squeeze(-1)  # [B, 1] -> [B]
            else:
                return yhat  # [B]
        else:
            # 多任务情况下，yhat是[B, task_count]，取第一个任务列
            return yhat[:, 0]  # [B, task_count] -> [B]





