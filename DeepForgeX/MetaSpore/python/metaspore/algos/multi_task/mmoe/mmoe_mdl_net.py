
import torch
import metaspore as ms
import sys

from ...layers import MLPLayer

class MdlMMoEModel(torch.nn.Module):
    """
    Modified MMoE for Multi-Domain Modeling.
    The original implementation treats tasks as different prediction heads (e.g., CTR, CVR).
    This version adapts it for multi-domain scenarios where each domain has its own data distribution
    but shares underlying patterns learned by experts.

    The key change is conceptualizing "task_numb" as "domain_numb" and potentially having
    a single tower output per domain (for binary classification like CTR) instead of multiple task outputs.
    However, the original structure can still be adapted by having one tower per domain.
    """
    def __init__(self,
                 embedding_dim=10,
                 column_name_path=None,
                 combine_schema_path=None,
                 expert_numb=6,
                 domain_numb=20,  # Renamed from task_numb to reflect domain modeling
                 expert_hidden_units=[256, 128],
                 expert_out_dim=10,
                 gate_hidden_units=[64],
                 tower_hidden_units=[64],
                 dnn_activations='ReLU',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 sparse_init_var=1e-2,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 tower_output_dim=1): # Added parameter for flexibility (e.g., 1 for CTR, 2 for multi-class)
        super().__init__()
        self.expert_numb = expert_numb
        self.domain_numb = domain_numb # Renamed
        self.expert_out_dim = expert_out_dim
        self.tower_output_dim = tower_output_dim

        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1 = ftrl_l1, \
                                                    l2 = ftrl_l2, \
                                                    alpha = ftrl_alpha, \
                                                    beta = ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.input_dim = int(self.sparse.feature_count*self.embedding_dim)

        # --- Experts: Shared across all domains ---
        self.experts = torch.nn.ModuleList() # Use ModuleList for proper registration
        for i in range(self.expert_numb):
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

        # --- Gates: One per domain ---
        self.gates = torch.nn.ModuleList() # Use ModuleList for proper registration
        for i in range(self.domain_numb):
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
        self.gate_softmax = torch.nn.Softmax(dim=1)

        # --- Towers: One per domain ---
        self.towers = torch.nn.ModuleList() # Use ModuleList for proper registration
        for i in range(self.domain_numb):
            mlp = MLPLayer(input_dim = self.expert_out_dim,
                           output_dim = self.tower_output_dim,
                           hidden_units = tower_hidden_units,
                           hidden_activations = dnn_activations,
                           final_activation = 'Sigmoid', # Assuming binary classification for CTR
                           dropout_rates = net_dropout,
                           input_norm=input_norm,
                           batch_norm = batch_norm,
                           use_bias = use_bias)
            self.towers.append(mlp)

        # --- Initialize internal attribute for domain IDs ---
        self._current_domain_ids = None


    def do_extra_work(self, minibatch):
        """
        在 model.py 中调用，做一些额外的工作（不方便在 forward 中进行），主要用于提取并缓存当前批次的 domain_id 信息。
        """
        # 假设 minibatch 是 pandas DataFrame 或 dict of tensors
        if isinstance(minibatch, dict):
            domain_id_col = minibatch['domain_id']
        elif hasattr(minibatch, 'domain_id'):  # 如 Spark Row 或 namedtuple
            domain_id_col = minibatch.domain_id
        else:
            # 如果是 pandas DataFrame
            domain_id_col = minibatch['domain_id'].values  # 或 .to_numpy()

        # 转为 LongTensor（PyTorch 推荐索引用 long/int64）
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()

        #print('do_extra_work-domain_id_tensor:', domain_id_tensor)
        self._current_domain_ids = domain_id_tensor  # 保存到模型内部

    def forward(self, x):
        x = self.sparse(x)
        batch_size = x.size(0)

        expert_outputs = [expert(x) for expert in self.experts]
        expert_cat = torch.stack(expert_outputs, dim=1)

        all_tower_outputs = []
        #计算每条样本在所有 domain 上的输出
        for i in range(self.domain_numb):
            gate_out = self.gates[i](x)
            gate_out = self.gate_softmax(gate_out).unsqueeze(-1)
            tower_input = torch.sum(expert_cat * gate_out, dim=1)
            tower_out = self.towers[i](tower_input)
            all_tower_outputs.append(tower_out)

        #最终只返回与当前样本 domain 对应的那个 tower 的输出
        if self._current_domain_ids is not None:
            domain_ids = self._current_domain_ids
            assert domain_ids.size(0) == batch_size
            
            stacked_outputs = torch.stack(all_tower_outputs, dim=1)
            expanded_ids = domain_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.tower_output_dim)
            selected_outputs = torch.gather(stacked_outputs, 1, expanded_ids).squeeze(1)
            self._current_domain_ids = None
            return selected_outputs
        else:
            return torch.cat(all_tower_outputs, dim=1)
    