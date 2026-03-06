import torch
import torch.nn as nn
import torch.nn.functional as F
import metaspore as ms
from ..layers import MLPLayer

##################### STAR HC2 模型 ######################
# 在 STAR 模型基础上增加H2C模块，用于生成域间对比损失
#########################################################

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

class HC2STARModel(torch.nn.Module):
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
                 # HC2 相关参数（默认关闭）
                 enable_hc2=False,
                 contrastive_weight_general: float = 0.1,
                 contrastive_weight_individual: float = 0.05,
                 temperature: float = 0.1,
                 diffusion_noise_scale: float = 0.1,
                 inverse_weight_temperature: float = 0.05,
                 dropout_augmentation_prob: float = 0.1,
                 **kwargs):
        super(HC2STARModel, self).__init__()
        self._current_domain_ids = None
        self.num_domains = num_domains
        self.sparse_embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.use_gate_nu = use_gate_nu
        self.star_fusion_activation_str = star_fusion_activation
        self.domain_net_init_bias = domain_net_init_bias
        
        # HC2 相关参数
        self.enable_hc2 = enable_hc2
        if self.enable_hc2:
            self.contrastive_weight_general = contrastive_weight_general
            self.contrastive_weight_individual = contrastive_weight_individual
            self.temperature = temperature
            self.diffusion_noise_scale = diffusion_noise_scale
            self.inverse_weight_temperature = inverse_weight_temperature
            self.dropout_augmentation_prob = dropout_augmentation_prob

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
            # 初始化最后一个线性层的偏置（适配你的 MLPLayer 结构）
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
        
        # HC2 相关组件
        if self.enable_hc2:
            # Dropout for augmentation in individual contrastive loss
            self.dropout_for_aug = nn.Dropout(p=self.dropout_augmentation_prob)

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
        h_domain_batch = torch.cat(h_domain_list, dim=0)  # (B, H)

        # 4. 优化的 STAR 星型拓扑融合：先应用激活函数再逐元素相乘
        h_domain_activated = self.star_fusion_activation(h_domain_batch)  # (B, H) 应用激活函数约束输出范围
        h_fused = h_center * h_domain_activated  # (B, H)

        # 5. 主预测头
        main_logits = self.final_mlp(h_fused)

        # 6. 辅助领域预测
        domain_embedded = self.domain_embedding(domain_ids)
        aux_logits = self.auxiliary_net(domain_embedded)

        # 7. 合并预测
        combined_logits = main_logits + aux_logits
        output = self.final_activation_layer(combined_logits) if self.final_activation_layer else combined_logits
        
        # 如果启用 HC2，返回中间表示用于对比学习
        if self.enable_hc2:
            return output, h_center, h_domain_batch, h_fused
        else:
            return output

    def compute_contrastive_losses(self, h_center, h_domain_batch, labels, domain_ids):
        """Compute both generalized and individual contrastive losses with enhanced features."""
        # Normalize embeddings for cosine similarity
        h_center_norm = F.normalize(h_center, p=2, dim=1)
        h_domain_norm = F.normalize(h_domain_batch, p=2, dim=1)
        
        # Generalized Contrastive Loss (based on shared representations and labels)
        gen_loss = self._compute_generalized_contrastive_loss_enhanced(h_center_norm, labels, domain_ids)
        
        # Individual Contrastive Loss (based on domain-specific representations and domains)
        ind_loss = self._compute_individual_contrastive_loss_enhanced(h_domain_norm, domain_ids)
        
        return gen_loss, ind_loss

    def _compute_generalized_contrastive_loss_enhanced(self, h_norm, labels, domain_ids):
        """
        Enhanced Generalized Contrastive Loss with:
        1. Diffusion noise for negative samples
        2. Inverse similarity weighting
        """
        batch_size = h_norm.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(h_norm, h_norm.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = label_mask.float()
        
        # Exclude self-similarity
        pos_mask = pos_mask - torch.eye(batch_size, device=pos_mask.device)
        
        # Negative mask (different labels)
        neg_mask = 1 - pos_mask
        
        # --- ENHANCEMENT 1: Diffusion Noise for Negative Samples ---
        h_with_noise = h_norm + torch.randn_like(h_norm) * self.diffusion_noise_scale
        h_with_noise_norm = F.normalize(h_with_noise, p=2, dim=1)
        sim_matrix_with_noise = torch.matmul(h_norm, h_with_noise_norm.T) / self.temperature
        
        # Use original similarity for positive pairs, noisy similarity for negative pairs
        sim_matrix_enhanced = sim_matrix * pos_mask + sim_matrix_with_noise * neg_mask
        
        # --- ENHANCEMENT 2: Inverse Similarity Weighting ---
        with torch.no_grad():
            # Compute inverse similarity weights based on original similarity
            inv_sim_weights = torch.exp(-sim_matrix / self.inverse_weight_temperature)
            # Normalize weights for each sample
            inv_sim_weights = inv_sim_weights / (inv_sim_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute log probabilities with enhanced similarities
        exp_sim = torch.exp(sim_matrix_enhanced)
        exp_sim_sum = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True) + torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        log_prob = sim_matrix_enhanced - torch.log(exp_sim_sum + 1e-8)
        
        # Apply inverse similarity weighting to positive samples
        weighted_pos_log_prob = (log_prob * pos_mask * inv_sim_weights).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        
        # Generalized contrastive loss with enhancements
        gen_loss = -weighted_pos_log_prob.mean()
        
        return gen_loss

    def _compute_individual_contrastive_loss_enhanced(self, h_norm, domain_ids):
        """
        Enhanced Individual Contrastive Loss based on:
        1. Scene-aware contrastive samples: augmented samples with dropout
        2. Cross-scene encoding strategy: using shared repr with other domain-specific nets
        """
        batch_size = h_norm.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(h_norm, h_norm.T) / self.temperature
        
        # Create mask for same domain (for positive samples in domain-specific context)
        domain_mask = domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1)
        pos_mask = domain_mask.float() - torch.eye(batch_size, device=domain_mask.device)
        
        # Negative mask (different domains)
        neg_mask = 1 - domain_mask.float()
        
        # --- ENHANCEMENT: Scene-aware contrastive samples using dropout augmentation ---
        # Create augmented representations by applying dropout to domain-specific representations
        h_aug = self.dropout_for_aug(h_norm)
        h_aug_norm = F.normalize(h_aug, p=2, dim=1)
        sim_matrix_aug = torch.matmul(h_norm, h_aug_norm.T) / self.temperature
        
        # For positive pairs, use both original and augmented similarities
        # For negative pairs, use original similarities
        sim_matrix_enhanced = sim_matrix * neg_mask + sim_matrix_aug * pos_mask
        
        # Compute log probabilities with enhanced similarities
        exp_sim = torch.exp(sim_matrix_enhanced)
        exp_sim_neg = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True)
        exp_sim_pos = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        
        # Log probability calculation
        denominator = exp_sim_pos + exp_sim_neg
        log_prob = sim_matrix_enhanced - torch.log(denominator + 1e-8)
        
        # Average log probabilities for same-domain (positive) samples
        pos_log_prob = (log_prob * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        
        # Individual contrastive loss (without diffusion noise and inverse similarity weighting)
        ind_loss = -pos_log_prob.mean()
        
        return ind_loss

    def compute_loss(self, predictions, labels, minibatch, **kwargs):
        """Compute total loss including contrastive losses if HC2 is enabled."""
        if self.enable_hc2:
            if isinstance(predictions, tuple):
                output, h_center, h_domain_batch, h_fused = predictions
            else:
                # 这种情况不应该发生
                raise ValueError("当启用HC2时，predictions应为(output, h_center, h_domain_batch, h_fused)元组")
                
            gen_loss, ind_loss = self.compute_contrastive_losses(h_center, h_domain_batch, labels, self._current_domain_ids)
            
            labels = labels.float().view(-1)
            output = output.view(-1)
            main_loss = F.binary_cross_entropy(output, labels)
            
            # 总损失
            total_loss = main_loss + \
                         self.contrastive_weight_general * gen_loss + \
                         self.contrastive_weight_individual * ind_loss
            
            return total_loss, main_loss
        else:
            # 未启用 HC2，只计算主损失
            output = predictions if not isinstance(predictions, tuple) else predictions[0]
            labels = labels.float().view(-1)
            output = output.view(-1)
            main_loss = F.binary_cross_entropy(output, labels)
            return main_loss, main_loss
    
    def predict(self, yhat, minibatch=None):
        """Prediction function for inference."""
        if self.enable_hc2:
            if isinstance(yhat, tuple):
                pvr, _, _, _ = yhat
            else:
                pvr = yhat
        else:
            pvr = yhat
        return pvr