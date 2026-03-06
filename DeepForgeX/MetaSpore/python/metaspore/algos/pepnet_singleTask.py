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

    def forward(self, x, domain_ids):
        """
        通过domain-specific信息对general input进行个性化调整
        Args:
            x: [batch_size, input_dim] - 来自Shared Bottom的通用表示
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
        personalized_output = x * gate_weights  # [B, D_SB]
        return personalized_output


class TaskTower(torch.nn.Module):
    """单任务塔"""
    def __init__(self, input_dim, hidden_units, output_dim=1):
        """
        Args:
            input_dim: 输入维度
            hidden_units: 隐藏层维度列表，如[128, 64]
            output_dim: 输出维度，默认为1
        """
        super(TaskTower, self).__init__()
        self.hidden_units = hidden_units

        # Create task tower layers
        layer_dims = [input_dim] + list(hidden_units) + [output_dim]
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # No activation after the last linear layer
                layers.append(torch.nn.ReLU())
        self.tower = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - 输入特征
        Returns:
            torch.Tensor: [batch_size, output_dim] - 预测logits
        """
        return self.tower(x)


class PEPNetSingleTask(torch.nn.Module):
    """单任务单场景的参数和嵌入个性化网络 (Parameter and Embedding Personalized Network) - With Shared Bottom MLP"""
    def __init__(self,
                 domain_count=1,
                 embedding_dim=16,
                 base_dnn_hidden_units=[512, 256], # Define the Shared Bottom MLP layers
                 task_tower_hidden_units=[128, 64],
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
                 ftrl_beta=1.0):
        super(PEPNetSingleTask, self).__init__()

        self.domain_count = domain_count
        self.embedding_dim = embedding_dim
        self.scale_factor = scale_factor

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
        shared_bottom_output_dim = base_dnn_hidden_units[-1] if base_dnn_hidden_units else shared_input_dim

        # --- Normalization for Shared Embedding ---
        self.bn_shared = ms.nn.Normalization(shared_input_dim)

        # --- Shared Bottom MLP ---
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
        self.ep_net = EPNetLayer(input_dim=shared_bottom_output_dim,
                                 domain_count=domain_count,
                                 embedding_dim=embedding_dim)

        # --- Task Tower ---
        self.task_tower = TaskTower(input_dim=shared_bottom_output_dim,
                                    hidden_units=task_tower_hidden_units,
                                    output_dim=1)

        # --- Final Activation ---
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Model forward pass. Computes prediction for single task.
        Assumes `do_extra_work` has been called to set `self._current_*` attributes.
        """
        # Retrieve pre-processed data (set by do_extra_work)
        domain_ids = self._current_domain_ids  # [B]
        # x is the input for shared embedding [B, N]

        # --- 1. Shared Embedding & BN ---
        shared_emb = self.shared_embedding(x)  # [B, D_shared]
        shared_emb_bn = self.bn_shared(shared_emb)    # [B, D_shared]

        # --- 2. Shared Bottom MLP ---
        shared_bottom_output = self.shared_bottom_layers(shared_emb_bn) # [B, D_SB]

        # --- 3. Prior Embedding ---
        prior_emb = self.prior_embedding(x) # [B, D_prior]

        # --- 4. EPNet: Embedding Personalization ---
        personalized_emb = self.ep_net(shared_bottom_output, domain_ids)  # [B, D_SB]

        # --- 5. Task Tower ---
        logits = self.task_tower(personalized_emb)  # [B, 1]

        # --- 6. Final Activation ---
        predictions = self.final_activation(logits)  # [B, 1]

        return predictions.squeeze(-1) # Return [B] tensor

    def do_extra_work(self, minibatch):
        """
        Pre-processes the minibatch to extract domain_id only.
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

    def predict(self, yhat, minibatch=None):
        # Return the prediction
        return yhat # [B]
