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

class CrossNetworkWithDomain(nn.Module):
    """
    带领域信息的交叉网络，结合SACN的交叉层设计
    """
    def __init__(self, input_dim, domain_dim, cross_layers=3):
        super(CrossNetworkWithDomain, self).__init__()
        self.cross_layers = nn.ModuleList()
        self.cross_biases = nn.ParameterList()
        
        # 总输入维度包括共享特征和领域嵌入
        total_input_dim = input_dim + domain_dim
        
        for i in range(cross_layers):
            # 每个交叉层学习一个1维投影来加权当前输出
            cross_layer = nn.Linear(total_input_dim, 1, bias=False)
            bias = nn.Parameter(torch.zeros(total_input_dim))
            self.cross_layers.append(cross_layer)
            self.cross_biases.append(bias)

    def forward(self, x_shared, x_domain):
        # 连接共享特征和领域特征
        x_total = torch.cat([x_shared, x_domain], dim=1)  # (B, shared_dim + domain_dim)
        x0 = x_total  # 保留原始输入用于交叉连接
        
        x_cross = x_total
        for cross_layer, bias in zip(self.cross_layers, self.cross_biases):
            # 计算交叉项: (W_l^T * x_l) * x_0 + b_l
            cross_proj = cross_layer(x_cross).unsqueeze(-1)  # (B, 1, 1)
            cross_term = (x0.unsqueeze(1) * cross_proj).squeeze(1)  # (B, total_dim)
            x_cross = cross_term + bias + x_cross  # 残差连接
            
        return x_cross

class StarCrossFusionModel(nn.Module):
    """
    改进版融合STAR和SACN优点的新型多领域推荐模型
    """
    def __init__(
        self,
        num_domains=1,
        embedding_dim=8,
        # --- Feature Columns ---
        column_name_path=None,
        combine_schema_path=None,
        # --- SACN Components ---
        cross_layers=3,
        # --- STAR Components ---
        center_hidden_units=[512, 256, 128],
        domain_specific_hidden_units=[512, 256, 128],
        # --- Final MLP ---
        final_hidden_units=[64],
        final_net_activation="sigmoid",
        # --- Network Config ---
        hidden_activations="ReLU",
        net_dropout=0,
        batch_norm=False,
        use_bias=True,
        # --- Embedding Config ---
        embedding_initializer=None,
        embedding_init_var=0.01,
        embedding_regularizer=None,
        net_regularizer=None,
        ftrl_l1=1.0,
        ftrl_l2=120.0,
        ftrl_alpha=0.5,
        ftrl_beta=1.0,
        # --- STAR Specific Config ---
        use_gate_nu=False,
        gate_scale_factor=1.0,
        star_fusion_activation="tanh",
        domain_net_init_bias=1.0,
        # --- Auxiliary Network Config ---
        auxiliary_hidden_units=[32],
        **kwargs
    ):
        super(StarCrossFusionModel, self).__init__()

        self.num_domains = num_domains
        self.sparse_embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.use_gate_nu = use_gate_nu
        self.star_fusion_activation_str = star_fusion_activation
        self.domain_net_init_bias = domain_net_init_bias

        # 共享稀疏嵌入层
        self.shared_sparse_embedding = ms.EmbeddingSumConcat(
            self.sparse_embedding_dim,
            self.column_name_path,
            self.combine_schema_path
        )
        if ftrl_l1 > 0 or ftrl_l2 > 0:
            self.shared_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        if embedding_initializer is None or embedding_initializer.lower() == "normal":
            self.shared_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)

        # 领域嵌入（用于辅助网络和交叉网络）
        self.domain_embedding = nn.Embedding(num_embeddings=self.num_domains, embedding_dim=self.sparse_embedding_dim)

        # 计算输入维度
        input_dim_for_nets = self.shared_sparse_embedding.feature_count * self.sparse_embedding_dim
        
        # 分区归一化层（来自STAR）
        self.partitioned_norm = PartitionedNormalization(self.num_domains, input_dim_for_nets)

        # GateNU 层（来自STAR，可选）
        if self.use_gate_nu:
            self.gate_nu = GateNULayer(input_dim_for_nets, scale_factor=gate_scale_factor)
        else:
            self.gate_nu = None

        # SACN风格的交叉网络：先处理processed_features
        self.cross_network = CrossNetworkWithDomain(
            input_dim=input_dim_for_nets,  # 输入原始处理后的特征
            domain_dim=self.sparse_embedding_dim,
            cross_layers=cross_layers
        )
        
        # 更新交叉网络输出的维度
        cross_output_dim = input_dim_for_nets + self.sparse_embedding_dim

        # 中心共享网络（现在接收交叉网络的输出）- 来自STAR
        # 修改中心网络的输入维度以匹配交叉网络输出
        center_input_dim = cross_output_dim
        center_hidden_units_with_input = [center_input_dim] + center_hidden_units
        center_output_dim = center_hidden_units[-1]
        
        self.center_net = MLPLayer(
            input_dim=center_input_dim,
            output_dim=center_output_dim,
            hidden_units=center_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )

        # 领域特定网络（现在也接收交叉网络的输出）- 来自STAR
        # 保证领域特定网络输出维度与中心网络一致
        assert center_output_dim == domain_specific_hidden_units[-1], \
            "Center and domain-specific networks must have the same output dimension for STAR fusion"
            
        domain_input_dim = cross_output_dim
        domain_specific_hidden_units_with_input = [domain_input_dim] + domain_specific_hidden_units
        domain_specific_output_dim = domain_specific_hidden_units[-1]
        
        self.domain_specific_nets = nn.ModuleList()
        for _ in range(self.num_domains):
            net = MLPLayer(
                input_dim=domain_input_dim,
                output_dim=domain_specific_output_dim,
                hidden_units=domain_specific_hidden_units,
                hidden_activations=hidden_activations,
                final_activation=None,
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
            self.star_fusion_activation = nn.Identity()

        # 最终预测MLP（作用于STAR融合后的特征）
        self.final_mlp = MLPLayer(
            input_dim=center_output_dim,  # 输入STAR融合后的特征维度
            output_dim=1,
            hidden_units=final_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )

        # 辅助网络：显式注入领域信息（来自STAR）
        aux_input_dim = self.sparse_embedding_dim
        aux_layers = []
        prev_dim = aux_input_dim
        for units in auxiliary_hidden_units:
            aux_layers.extend([
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.Dropout(net_dropout) if net_dropout > 0 else nn.Identity()
            ])
            prev_dim = units
        aux_layers.append(nn.Linear(prev_dim, 1))
        self.auxiliary_net = nn.Sequential(*aux_layers)

        # 最终激活函数
        self.final_activation_layer = nn.Sigmoid() if final_net_activation == "sigmoid" else None

        # 存储当前批次的领域ID
        self._current_domain_ids = None

    def do_extra_work(self, minibatch):
        # 提取 domain_id 并保存
        if isinstance(minibatch, dict):
            domain_id_col = minibatch.get('domain_id')
        elif hasattr(minibatch, 'domain_id'):
            domain_id_col = minibatch.domain_id
        else:
            raise ValueError("Input batch must contain 'domain_id' column.")
        
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()
        self._current_domain_ids = domain_id_tensor

    def forward(self, x):
        # 共享嵌入特征
        embedded_features = self.shared_sparse_embedding(x)  # (B, F * E)
        batch_size = embedded_features.shape[0]
        domain_ids = self._current_domain_ids
        assert domain_ids is not None and domain_ids.shape[0] == batch_size

        # 1. 分区归一化（来自STAR）
        pn_normalized_features = self.partitioned_norm(embedded_features, domain_ids)  # (B, F * E)

        # 2. GateNU 门控（来自STAR，可选）
        if self.use_gate_nu:
            processed_features = self.gate_nu(pn_normalized_features)  # (B, F * E)
        else:
            processed_features = pn_normalized_features  # (B, F * E)

        # 3. SACN交叉网络：处理经过GateNU处理后的特征
        domain_embedded = self.domain_embedding(domain_ids)  # (B, E)
        h_cross = self.cross_network(processed_features, domain_embedded)  # (B, F*E + E)

        # 4. STAR架构：中心网络和领域特定网络（现在接收交叉网络输出）
        h_center = self.center_net(h_cross)  # (B, H)

        h_domain_list = []
        for i in range(batch_size):
            d_id = domain_ids[i].item()
            d_out = self.domain_specific_nets[d_id](h_cross[i:i+1, :])  # (1, H) 接收交叉网络输出
            h_domain_list.append(d_out)
        h_domain = torch.cat(h_domain_list, dim=0)  # (B, H)

        # 5. STAR融合：先应用激活函数再逐元素相乘
        h_domain_activated = self.star_fusion_activation(h_domain)  # (B, H)
        h_star_fused = h_center * h_domain_activated  # (B, H)

        # 6. 主预测头
        main_logits = self.final_mlp(h_star_fused)  # (B, 1)

        # 7. 辅助领域预测（来自STAR）
        aux_logits = self.auxiliary_net(domain_embedded)  # (B, 1)

        # 8. 合并主预测和辅助预测
        combined_logits = main_logits + aux_logits
        output = self.final_activation_layer(combined_logits) if self.final_activation_layer else combined_logits
        return output
