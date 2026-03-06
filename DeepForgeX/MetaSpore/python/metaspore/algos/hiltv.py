import torch
import metaspore as ms
from torch.nn import Parameter
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class ResidualExpertBlock(torch.nn.Module):
    """残差专家块 (REB)"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ResidualExpertBlock, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, input_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + residual  # 残差连接
        return out


class DualAttentionBlock(torch.nn.Module):
    """双注意力块 (DAB)"""
    def __init__(self, feature_dim: int, num_experts: int):
        super(DualAttentionBlock, self).__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        
        # 特征级注意力
        self.feature_attention = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * num_experts, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, feature_dim * num_experts),
            torch.nn.Sigmoid()
        )
        
        # 专家级注意力
        self.expert_attention = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * num_experts, num_experts),
            torch.nn.ReLU(),
            torch.nn.Linear(num_experts, num_experts * feature_dim),
            torch.nn.Sigmoid()
        )
        
    def forward(self, expert_outputs):
        B, M, C = expert_outputs.shape  # Batch, Feature Dim, Num Experts
        
        # 重塑为适合注意力计算的形状
        reshaped = expert_outputs.permute(0, 2, 1).contiguous().view(B, -1)  # [B, M*C]
        
        # 特征级注意力
        feat_attn = self.feature_attention(reshaped).view(B, C, M).permute(0, 2, 1)  # [B, M, C]
        
        # 专家级注意力
        exp_attn = self.expert_attention(reshaped).view(B, M, C)
        
        # 应用注意力权重
        attended = expert_outputs * feat_attn * exp_attn
        
        # 降维回原始特征维度
        output = torch.sum(attended, dim=2)  # [B, M]
        
        return output


class SubDistributionLearningModule(torch.nn.Module):
    """子分布学习模块 (SLM)"""
    def __init__(self, input_dim: int, num_sub_distributions: int):
        super(SubDistributionLearningModule, self).__init__()
        self.num_sub_distributions = num_sub_distributions
        
        # 购买概率预测塔
        self.purchase_prob_tower = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim // 2, 1),
            torch.nn.Sigmoid()
        )
        
        # 子分布参数预测塔 (均值、标准差、权重)
        self.sub_dist_params_tower = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(input_dim // 2, 3)  # mu, sigma, weight
            ) for _ in range(num_sub_distributions)
        ])
        
    def forward(self, features):
        # 购买概率
        purchase_prob = self.purchase_prob_tower(features)
        
        # 子分布参数
        sub_dist_params = []
        for i, tower in enumerate(self.sub_dist_params_tower):
            params = tower(features)  # [batch_size, 3] -> [mu, sigma, weight]
            mu = params[:, 0]
            sigma = torch.clamp(params[:, 1], min=1e-6)  # 防止sigma为0
            weight_raw = params[:, 2]
            # 使用softmax确保权重归一化
            weights_all = torch.stack([tower(features)[:, 2] for tower in self.sub_dist_params_tower], dim=1)
            weight = torch.softmax(weights_all, dim=1)[:, i]
            
            sub_dist_params.append((mu, sigma, weight))
            
        return purchase_prob, sub_dist_params


class NewUserCalibrationModule(torch.nn.Module):
    """新用户校准模块 (NUCM)"""
    def __init__(self, feature_dim: int):
        super(NewUserCalibrationModule, self).__init__()
        self.feature_dim = feature_dim
        
        # 用于生成仿射变换参数的MLP
        self.calibration_mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim // 2),
            torch.nn.BatchNorm1d(feature_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim // 2, feature_dim * 2)  # 生成W_a和b_a
        )
        
        # 可学习的旋转矩阵
        self.rotation_matrix = Parameter(torch.randn(feature_dim, feature_dim))
        
    def forward(self, features, is_new_user):
        # 生成仿射变换参数
        affine_params = self.calibration_mlp(features.detach())  # 使用detach停止梯度
        W_a = affine_params[:, :self.feature_dim].unsqueeze(1)  # [B, 1, feature_dim]
        b_a = affine_params[:, self.feature_dim:].unsqueeze(1)  # [B, 1, feature_dim]
        
        # 旋转矩阵
        R = F.normalize(self.rotation_matrix, p=2, dim=1)  # 归一化
        
        # 应用校准
        if is_new_user.any():
            # 对新用户应用校准变换
            transformed_features = torch.bmm(W_a * features.unsqueeze(1) + b_a, R.unsqueeze(0))  # [B, 1, feature_dim]
            return transformed_features.squeeze(1)
        else:
            return features


class HiLTV(torch.nn.Module):
    """
    HiLTV: 用于在线游戏生命周期价值预测的层次多分布建模
    """
    def __init__(self, 
                 embedding_dim: int = 16,
                 column_name_path: str = None,
                 combine_schema_path: str = None,
                 num_experts: int = 5,
                 hidden_dim: int = 64,
                 dnn_hidden_units: list = [512, 256, 128],
                 dnn_hidden_activations="ReLU",
                 net_dropout=0.1,
                 batch_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_bias=True,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 final_net_activation="linear",
                 **kwargs):
        super(HiLTV, self).__init__()
        
        # 初始化稀疏嵌入层
        self._embedding_dim = embedding_dim
        self._sparse = ms.EmbeddingSumConcat(
            self._embedding_dim,
            column_name_path,
            combine_schema_path
        )
        self._sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, 
            l2=ftrl_l2, 
            alpha=ftrl_alpha, 
            beta=ftrl_beta
        )
        self._sparse.initializer = ms.NormalTensorInitializer(var=0.001)
        
        extra_attributes = {
            "enable_fresh_random_keep": True,
            "fresh_dist_range_from": 0,
            "fresh_dist_range_to": 1000,
            "fresh_dist_range_mean": 950,
        }
        self._sparse.extra_attributes = extra_attributes
        
        feature_count = self._sparse.feature_count
        feature_dim = self._sparse.feature_count * self._embedding_dim
        
        # 批归一化层
        if batch_norm:
            self._bn = ms.nn.Normalization(feature_dim, momentum=0.01, eps=1e-5, affine=True)
        else:
            self._bn = None
            
        # 共享隐藏层
        self._shared_layers = torch.nn.Sequential()
        prev_dim = feature_dim
        for i, unit in enumerate(dnn_hidden_units):
            layer = torch.nn.Linear(prev_dim, unit)
            activation = torch.nn.ReLU() if dnn_hidden_activations == "ReLU" else torch.nn.Tanh()
            dropout_layer = torch.nn.Dropout(net_dropout) if net_dropout > 0 else torch.nn.Identity()
            
            self._shared_layers.add_module(f"linear_{i}", layer)
            self._shared_layers.add_module(f"activation_{i}", activation)
            if net_dropout > 0:
                self._shared_layers.add_module(f"dropout_{i}", dropout_layer)
            
            prev_dim = unit
        
        # 层次化支付偏好感知模块 (HP3M)
        self.hp3m = torch.nn.ModuleList([
            ResidualExpertBlock(prev_dim, prev_dim // 2) for _ in range(num_experts)
        ])
        self.dab = DualAttentionBlock(prev_dim, num_experts)
        
        # 子分布学习模块 (SLM)
        self.slm = SubDistributionLearningModule(prev_dim, num_experts)
        
        # 新用户校准模块 (NUCM)
        self.nucm = NewUserCalibrationModule(prev_dim)
        
        # 最终输出层
        self._output_layer = torch.nn.Linear(prev_dim, 1)
        
        # 激活函数
        if final_net_activation == 'sigmoid':
            self._final_activation = torch.nn.Sigmoid()
        elif final_net_activation == 'relu':
            self._final_activation = torch.nn.ReLU()
        else:
            self._final_activation = torch.nn.Identity()
        
        # 保存配置
        self.num_experts = num_experts
        self._feature_dim = feature_dim

    def forward(self, x):
        # x 包含所有特征 (用户特征 + 游戏特征 + 新用户标志)
        
        # 稀疏嵌入
        embedded = self._sparse(x)
        
        # 批归一化
        if self._bn is not None:
            embedded = self._bn(embedded)
        
        # 共享隐藏层
        shared_output = self._shared_layers(embedded)
        
        # 提取新用户标志 (假设最后一维是新用户标志)
        # 这里需要根据实际输入数据结构调整
        if hasattr(self, '_new_user_flag_idx'):
            is_new_user = x[:, self._new_user_flag_idx].bool()
        else:
            # 默认假设新用户标志在嵌入的最后一维
            is_new_user = torch.zeros(embedded.size(0), dtype=torch.bool, device=embedded.device)
        
        # 通过各个专家
        expert_outputs = []
        for expert in self.hp3m:
            expert_out = expert(shared_output)
            expert_outputs.append(expert_out.unsqueeze(2))  # 添加专家维度
        
        # 拼接专家输出
        expert_outputs = torch.cat(expert_outputs, dim=2)  # [B, hidden_dim, num_experts]
        
        # 双注意力块
        attention_output = self.dab(expert_outputs)
        
        # 应用新用户校准
        calibrated_features = self.nucm(attention_output, is_new_user)
        
        # 子分布学习
        purchase_prob, sub_dist_params = self.slm(calibrated_features)
        
        # 最终输出
        final_output = self._output_layer(calibrated_features)
        final_output = self._final_activation(final_output)
        
        # 结合购买概率
        ltv_prediction = purchase_prob * final_output
        
        return ltv_prediction, purchase_prob, sub_dist_params


# ZIMoL (零膨胀混合逻辑回归) 损失函数
class ZIMoL_Loss(torch.nn.Module):
    """
    Zero-Inflated Mixture-of-Logistic Loss
    用于处理多模态LTV分布的损失函数
    """
    def __init__(self, num_components: int = 3, discretization_param: float = 0.1):
        super(ZIMoL_Loss, self).__init__()
        self.num_components = num_components
        self.discretization_param = discretization_param
        
    def logistic_cdf(self, x, mu, sigma):
        """逻辑分布的累积分布函数"""
        z = (x - mu) / (sigma + 1e-8)
        return torch.sigmoid(z)
    
    def forward(self, predicted_purchase_prob, sub_dist_params, true_values):
        batch_size = true_values.size(0)
        
        # 购买概率损失 (二分类交叉熵)
        purchase_targets = (true_values > 0).float()
        purchase_loss = torch.nn.functional.binary_cross_entropy(
            predicted_purchase_prob.squeeze(), 
            purchase_targets
        )
        
        # 只对付费用户计算LTV损失
        paid_mask = (true_values > 0)
        if not paid_mask.any():
            return purchase_loss
        
        paid_true_values = true_values[paid_mask]
        paid_indices = torch.nonzero(paid_mask).squeeze()
        
        if len(paid_indices.shape) == 0:  # 如果只有一个付费用户
            paid_indices = paid_indices.unsqueeze(0)
        
        # 获取付费用户的子分布参数
        paid_sub_params = []
        for params in sub_dist_params:
            mu = params[0][paid_mask]
            sigma = params[1][paid_mask]
            weight = params[2][paid_mask]
            paid_sub_params.append((mu, sigma, weight))
        
        # 计算逻辑混合分布的似然
        log_likelihoods = []
        for i in range(len(paid_true_values)):
            likelihood = 0
            for k in range(min(self.num_components, len(paid_sub_params))):
                mu_k = paid_sub_params[k][0][i]
                sigma_k = paid_sub_params[k][1][i]
                weight_k = paid_sub_params[k][2][i]
                
                # 计算在u处的概率密度
                cdf_upper = self.logistic_cdf(
                    paid_true_values[i] + self.discretization_param/2, 
                    mu_k, sigma_k
                )
                cdf_lower = self.logistic_cdf(
                    paid_true_values[i] - self.discretization_param/2, 
                    mu_k, sigma_k
                )
                prob_density = torch.clamp(cdf_upper - cdf_lower, min=1e-8)
                
                likelihood += weight_k * prob_density
            
            log_likelihood = torch.log(likelihood + 1e-8)
            log_likelihoods.append(log_likelihood)
        
        if len(log_likelihoods) > 0:
            ltv_loss = -torch.mean(torch.stack(log_likelihoods))
        else:
            ltv_loss = torch.tensor(0.0, requires_grad=True)
        
        # 总损失
        total_loss = purchase_loss + ltv_loss
        
        return total_loss


# 辅助函数：创建示例模型
def create_hilvt_model(column_name_path=None, combine_schema_path=None, **kwargs):
    """
    创建HiLTV模型的便捷函数
    """
    model = HiLTV(
        column_name_path=column_name_path,
        combine_schema_path=combine_schema_path,
        **kwargs
    )
    return model
