import torch
import torch.nn as nn
import metaspore as ms
from ..layers import MLPLayer

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

class PartitionedNormalization(nn.Module):
    """
    分区归一化层，为每个领域维护独立的归一化参数
    作用于共享 embedding 的输出
    """
    def __init__(self, num_domains, normalized_shape):
        super(PartitionedNormalization, self).__init__()
        self.num_domains = num_domains
        self.normalized_shape = normalized_shape
        
        # 为每个领域维护独立的 gamma 和 beta 参数
        self.weight = nn.Parameter(torch.ones(num_domains, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(num_domains, normalized_shape))
        
        # 运行时统计量
        self.register_buffer('running_mean', torch.zeros(num_domains, normalized_shape))
        self.register_buffer('running_var', torch.ones(num_domains, normalized_shape))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x, domain_ids):
        """
        x: (batch_size, feature_dim)
        domain_ids: (batch_size,)
        """
        batch_size = x.size(0)
        normalized_features = []
        
        for i in range(batch_size):
            d_id = domain_ids[i].item()
            single_feature = x[i:i+1, :]  # (1, feature_dim)
            
            # 获取对应领域的参数
            weight_d = self.weight[d_id:d_id+1, :]
            bias_d = self.bias[d_id:d_id+1, :]
            
            # 归一化（使用当前样本自身统计量，简化实现）
            mean = single_feature.mean(dim=1, keepdim=True)
            var = single_feature.var(dim=1, keepdim=True, unbiased=False)
            normalized = (single_feature - mean) / torch.sqrt(var + 1e-5)
            
            # 应用领域特定的仿射变换
            normalized = normalized * weight_d + bias_d
            normalized_features.append(normalized)
        
        return torch.cat(normalized_features, dim=0)

def _get_last_linear_layer(module):
    """从 torch.nn.Sequential 中提取最后一个 Linear 层"""
    linear_layers = []
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            linear_layers.append(layer)
    return linear_layers[-1] if linear_layers else None
    

class STARModel(torch.nn.Module):
    def __init__(self,
                 num_domains=1,
                 embedding_dim=8,
                 center_hidden_units=[512, 256, 128],
                 domain_specific_hidden_units=[512, 256, 128],  # 必须与 center 输出维度一致
                 final_hidden_units=[64],
                 column_name_path=None,
                 combine_schema_path=None,
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 use_bias=True,
                 embedding_initializer=None,
                 embedding_init_var=0.01,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 final_net_activation="sigmoid",
                 use_gate_nu=False,  # 新增：是否使用 GateNU
                 gate_scale_factor=1.0,
                 star_fusion_activation="tanh", # 新增：STAR融合前的激活函数
                 domain_net_init_bias=1.0,       # 新增：领域网络最后偏置初始化值
                 **kwargs):
        super(STARModel, self).__init__()
        self._current_domain_ids = None
        self.num_domains = num_domains
        self.sparse_embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.use_gate_nu = use_gate_nu
        self.star_fusion_activation_str = star_fusion_activation
        self.domain_net_init_bias = domain_net_init_bias

        # 共享嵌入层
        self.shared_sparse_embedding = ms.EmbeddingSumConcat(
            self.sparse_embedding_dim,
            column_name_path,
            combine_schema_path
        )
        if ftrl_l1 > 0 or ftrl_l2 > 0:
            self.shared_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        if embedding_initializer is None or embedding_initializer.lower() == "normal":
            self.shared_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)

        # 领域嵌入（用于辅助网络）
        self.domain_embedding = nn.Embedding(num_embeddings=self.num_domains, embedding_dim=self.sparse_embedding_dim)

        # 计算输入维度
        input_dim_for_nets = self.shared_sparse_embedding.feature_count * self.sparse_embedding_dim
        
        # 分区归一化层（作用于共享 embedding 的输出）
        self.partitioned_norm = PartitionedNormalization(self.num_domains, input_dim_for_nets)

        # GateNU 层（可选）
        if self.use_gate_nu:
            self.gate_nu = GateNULayer(input_dim_for_nets, scale_factor=gate_scale_factor)
        else:
            self.gate_nu = None

        # 中心共享网络（输出维度必须与领域特定网络一致）
        assert center_hidden_units[-1] == domain_specific_hidden_units[-1], \
            "Center and domain-specific networks must have the same output dimension for STAR fusion"
        shared_output_dim = center_hidden_units[-1]
        
        self.center_net = MLPLayer(
            input_dim=input_dim_for_nets,
            output_dim=shared_output_dim,
            hidden_units=center_hidden_units[:-1],
            hidden_activations=hidden_activations,
            final_activation=None, # 不使用最终激活，保持原始输出
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )

        # 领域特定网络（每个领域一个，输出维度与 center_net 一致）
        self.domain_specific_nets = nn.ModuleList()
        for _ in range(self.num_domains):
            net = MLPLayer(
                input_dim=input_dim_for_nets,
                output_dim=shared_output_dim,  # 关键：与 center_net 输出维度相同
                hidden_units=domain_specific_hidden_units[:-1],
                hidden_activations=hidden_activations,
                final_activation=None, # 不使用最终激活，由 self.star_fusion_activation 处理
                dropout_rates=net_dropout,
                batch_norm=False,
                use_bias=use_bias,
                input_norm=False,
            )
            # 初始化最后一个线性层的偏置
            last_linear = _get_last_linear_layer(net.dnn)
            if last_linear is not None:
                nn.init.constant_(last_linear.bias, self.domain_net_init_bias)
            self.domain_specific_nets.append(net)

        # STAR 融合激活函数
        if self.star_fusion_activation_str == "tanh":
            self.star_fusion_activation = nn.Tanh()
        elif self.star_fusion_activation_str == "sigmoid":
            self.star_fusion_activation = nn.Sigmoid()
        elif self.star_fusion_activation_str == "relu":
            self.star_fusion_activation = nn.ReLU()
        else:
            self.star_fusion_activation = nn.Identity() # 不使用额外激活

        # 最终预测MLP（作用于星型融合后的特征）
        self.final_mlp = MLPLayer(
            input_dim=shared_output_dim,
            output_dim=1,
            hidden_units=final_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )

        # 辅助网络：显式注入领域信息
        self.auxiliary_net = nn.Sequential(
            nn.Linear(self.sparse_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 最终激活函数
        self.final_activation_layer = nn.Sigmoid() if final_net_activation == "sigmoid" else None

    def do_extra_work(self, minibatch):
        # 提取 domain_id 并保存
        if isinstance(minibatch, dict):
            domain_id_col = minibatch['domain_id']
        elif hasattr(minibatch, 'domain_id'):
            domain_id_col = minibatch.domain_id
        else:
            domain_id_col = minibatch['domain_id'].values
        
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()
        self._current_domain_ids = domain_id_tensor

    def forward(self, x):
        embedded_features = self.shared_sparse_embedding(x) # (B, F * E)
        batch_size = embedded_features.shape[0]
        domain_ids = self._current_domain_ids
        assert domain_ids is not None and domain_ids.shape[0] == batch_size

        # 1. 分区归一化（作用于共享 embedding 输出）
        pn_normalized_features = self.partitioned_norm(embedded_features, domain_ids) # (B, F * E)

        # 2. GateNU 门控（可选）
        if self.use_gate_nu:
            processed_features = self.gate_nu(pn_normalized_features) # (B, F * E)
        else:
            processed_features = pn_normalized_features # (B, F * E)

        # 3. 中心网络和领域特定网络
        h_center = self.center_net(processed_features)  # (B, H)

        h_domain_list = []
        for i in range(batch_size):
            d_id = domain_ids[i].item()
            d_out = self.domain_specific_nets[d_id](processed_features[i:i+1, :])  # (1, H)
            h_domain_list.append(d_out)
        h_domain = torch.cat(h_domain_list, dim=0)  # (B, H)

        # 4. 优化的 STAR 星型拓扑融合：先应用激活函数再逐元素相乘
        h_domain_activated = self.star_fusion_activation(h_domain)  # (B, H) 应用激活函数约束输出范围
        h_fused = h_center * h_domain_activated  # (B, H)

        # 5. 主预测头
        main_logits = self.final_mlp(h_fused)

        # 6. 辅助领域预测
        domain_embedded = self.domain_embedding(domain_ids)
        aux_logits = self.auxiliary_net(domain_embedded)

        # 7. 合并预测
        combined_logits = main_logits + aux_logits
        output = self.final_activation_layer(combined_logits) if self.final_activation_layer else combined_logits
        return output
