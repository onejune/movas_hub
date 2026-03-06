import torch
import torch.nn.functional as F
import metaspore as ms
from ..layers import MLPLayer

class ExpertNetwork(torch.nn.Module):
    """专家网络（简化版）"""
    def __init__(self, input_dim, expert_units, activation='ReLU', dropout_rate=None):
        super().__init__()
        self.mlp = MLPLayer(
            input_dim=input_dim,
            hidden_units=expert_units,
            hidden_activations=activation,
            final_activation=activation,
            dropout_rates=dropout_rate,
            batch_norm=False  # 移除批归一化提升速度
        )

    def forward(self, x):
        return self.mlp(x)

class PLEGate(torch.nn.Module):
    """简化门控机制（移除top-k）"""
    def __init__(self, input_dim, num_experts, gate_hidden_units=None):
        super().__init__()
        gate_units = gate_hidden_units or [64]
        self.gate_mlp = MLPLayer(
            input_dim=input_dim,
            output_dim=num_experts,
            hidden_units=gate_units,
            final_activation=None
        )

    def forward(self, x):
        gate_weights = self.gate_mlp(x)
        return F.softmax(gate_weights, dim=-1)  # 简化softmax

class PLELayer(torch.nn.Module):
    """高效PLE层（向量化计算）"""
    def __init__(self, input_dim, domain_count, shared_expert_num=4,
                 domain_expert_num=2, expert_units=[256,128], gate_hidden_units=[64]):
        super().__init__()
        self.domain_count = domain_count
        
        # 共享专家
        self.shared_experts = torch.nn.ModuleList([
            ExpertNetwork(input_dim, expert_units) 
            for _ in range(shared_expert_num)
        ])
        
        # 领域专家（向量化存储）
        self.domain_experts = torch.nn.ModuleList([
            ExpertNetwork(input_dim, expert_units)
            for _ in range(domain_count * domain_expert_num)
        ])
        
        # 门控网络
        self.shared_gate = PLEGate(input_dim, shared_expert_num, gate_hidden_units)
        self.domain_gates = torch.nn.ModuleList([
            PLEGate(input_dim, shared_expert_num + domain_expert_num, gate_hidden_units)
            for _ in range(domain_count)
        ])

    def forward(self, x, domain_ids):
        # 批量计算共享专家输出 [B, S, E_out]
        shared_outputs = torch.stack([expert(x) for expert in self.shared_experts], dim=1)
        
        # 批量计算所有领域专家输出 [B, D*DE, E_out]
        all_domain_outputs = torch.stack([expert(x) for expert in self.domain_experts], dim=1)
        
        # 按领域分组 [D, B, DE, E_out]
        domain_outputs = all_domain_outputs.view(
            x.size(0), self.domain_count, -1, shared_outputs.size(-1)
        ).permute(1,0,2,3)
        
        # 计算每个领域的门控权重
        domain_results = []
        for d in range(self.domain_count):
            # 拼接共享和领域专家 [B, (S+DE), E_out]
            concat_experts = torch.cat([
                shared_outputs, 
                domain_outputs[d]
            ], dim=1)
            
            # 计算门控权重 [B, S+DE]
            gate_weights = self.domain_gates[d](x)
            
            # 加权求和 [B, E_out]
            weighted_output = torch.sum(
                concat_experts * gate_weights.unsqueeze(-1), 
                dim=1
            )
            domain_results.append(weighted_output)
        
        return domain_results  # List of [B, E_out]

class ExplicitLabelSpace(torch.nn.Module):
    """高效标签空间建模（梯度隔离）"""
    def __init__(self, domain_count, input_dim, gate_hidden_units=[64]):
        super().__init__()
        self.domain_count = domain_count
        self.gate = PLEGate(input_dim, domain_count, gate_hidden_units)

    def forward(self, domain_outputs, x, domain_ids):
        """
        domain_outputs: 各场景输出 [domain_count, B, feat_dim]
        domain_ids: 样本所属场景ID [B]
        """
        stacked_outputs = torch.stack(domain_outputs, dim=0)  # [D, B, F]
        
        # 计算场景权重 [B, D]
        gate_weights = self.gate(x)
        
        # 梯度隔离：仅当前场景保留梯度
        with torch.no_grad():
            detached_outputs = stacked_outputs.detach()
        
        # 重建梯度：仅当前场景的梯度保留
        batch_idx = torch.arange(stacked_outputs.size(1))
        detached_outputs[domain_ids, batch_idx] = stacked_outputs[domain_ids, batch_idx]
        
        # 加权融合 [B, F]
        return torch.sum(gate_weights.unsqueeze(-1) * detached_outputs.permute(1,0,2), dim=1)

class HMoELayer(torch.nn.Module):
    """高效HMoE层"""
    def __init__(self, input_dim, domain_count, shared_expert_num=4,
                 domain_expert_num=2, expert_units=[256,128], gate_hidden_units=[64]):
        super().__init__()
        self.ple = PLELayer(
            input_dim, domain_count, shared_expert_num, 
            domain_expert_num, expert_units, gate_hidden_units
        )
        self.label_space = ExplicitLabelSpace(
            domain_count, input_dim, gate_hidden_units
        )

    def forward(self, x, domain_ids):
        domain_outputs = self.ple(x, domain_ids)  # List of [B, E_out]
        return self.label_space(domain_outputs, x, domain_ids)  # [B, E_out]

class HMoENetwork(torch.nn.Module):
    """优化后的HMoE网络"""
    def __init__(self, domain_count=10, embedding_dim=8, shared_expert_num=2,
                 domain_expert_num=2, expert_units=[128,64], gate_hidden_units=[32],
                 column_name_path=None, combine_schema_path=None, 
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 sparse_init_var=1e-2):
        super().__init__()
        self.domain_count = domain_count
        
        # 共享嵌入层
        self.shared_embedding = ms.EmbeddingSumConcat(
            embedding_dim, column_name_path, combine_schema_path
        )
        self.shared_embedding.updater = ms.FTRLTensorUpdater(l1 = ftrl_l1, l2 = ftrl_l2, alpha = ftrl_alpha, beta = ftrl_beta)
        self.shared_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # 输入维度计算
        input_dim = self.shared_embedding.feature_count * embedding_dim
        
        # HMoE层
        self.hmoe_layer = HMoELayer(
            input_dim, domain_count, shared_expert_num,
            domain_expert_num, expert_units, gate_hidden_units
        )
        
        # 领域特定塔（稀疏梯度实现）
        self.domain_towers = torch.nn.ModuleList([
            torch.nn.Linear(expert_units[-1], 1)  # 简化为单层
            for _ in range(domain_count)
        ])
        
        # 最终激活函数
        self.final_activation = torch.nn.Sigmoid()
        
        # 存储当前domain_id
        self._current_domain_ids = None

    def forward(self, x):
        # 获取domain_id
        domain_ids = self._current_domain_ids
        
        # 共享嵌入
        shared_emb = self.shared_embedding(x)
        
        # HMoE处理
        hmoe_output = self.hmoe_layer(shared_emb, domain_ids)
        
        # 稀疏梯度预测
        batch_size = hmoe_output.size(0)
        predictions = torch.zeros(batch_size, device=hmoe_output.device)
        
        for d in range(self.domain_count):
            mask = (domain_ids == d)
            if mask.any():
                # 仅当前领域计算梯度
                predictions[mask] = self.domain_towers[d](hmoe_output[mask]).squeeze()
        
        return self.final_activation(predictions)

    def do_extra_work(self, minibatch):
        """处理额外的工作，如获取domain_id"""
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

        # 不再处理task_id，只处理domain_id
        self._current_domain_ids = domain_id_tensor
