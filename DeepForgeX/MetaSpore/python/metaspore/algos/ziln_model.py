import torch
import metaspore as ms
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from .layers import MLPLayer

class ZILNModel(torch.nn.Module):
    def __init__(self,
                use_wide=True,
                wide_embedding_dim=10,
                deep_embedding_dim=10,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                deep_column_name_path=None,
                deep_combine_schema_path=None,
                dnn_hidden_units=[512,256,64],
                dnn_hidden_activations="ReLU",
                net_dropout=0,
                batch_norm=False,
                embedding_regularizer=None,
                net_regularizer=None,
                use_bias=True,
                ftrl_l1=1.0,
                ftrl_l2=120.0,
                ftrl_alpha=0.5,
                ftrl_beta=1.0,
                **kwargs):
        super().__init__()
        self.use_wide = use_wide
        
        # Wide部分：线性特征（用于零膨胀部分）
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim,
                                                wide_column_name_path,
                                                wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # Deep部分：嵌入层（共享特征）
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim,
                                           deep_column_name_path,
                                           deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # 共享主干网络
        shared_input_dim = self.dnn_sparse.feature_count * deep_embedding_dim
        if self.use_wide:
            shared_input_dim += self.lr_sparse.feature_count * wide_embedding_dim
        
        self.shared_dnn = MLPLayer(input_dim=shared_input_dim,
                                output_dim=dnn_hidden_units[-1],
                                hidden_units=dnn_hidden_units[:-1],
                                hidden_activations=dnn_hidden_activations,
                                final_activation=dnn_hidden_activations,
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm,
                                use_bias=use_bias,
                                input_norm=True)
        
        # 零膨胀部分输出头：logit(pi) - 转化概率
        self.pi_head = MLPLayer(input_dim=dnn_hidden_units[-1],
                               output_dim=1,
                               hidden_units=[16],
                               hidden_activations=dnn_hidden_activations,
                               final_activation=None,
                               dropout_rates=net_dropout,
                               batch_norm=batch_norm,
                               use_bias=use_bias)
        
        # 正值部分输出头：均值μ
        self.mu_head = MLPLayer(input_dim=dnn_hidden_units[-1],
                               output_dim=1,
                               hidden_units=[16],
                               hidden_activations=dnn_hidden_activations,
                               final_activation=None,
                               dropout_rates=net_dropout,
                               batch_norm=batch_norm,
                               use_bias=use_bias)
        
        # 正值部分输出头：对数标准差log(σ)
        self.log_sigma_head = MLPLayer(input_dim=dnn_hidden_units[-1],
                                      output_dim=1,
                                      hidden_units=[16],
                                      hidden_activations=dnn_hidden_activations,
                                      final_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm,
                                      use_bias=use_bias)

    def forward(self, x):
        """
        前向传播，输出ZILN分布参数
        返回: (logit_pi, mu, log_sigma)
        """
        # 获取deep特征嵌入
        dnn_out = self.dnn_sparse(x)
        
        # Wide部分（如果启用）
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = wide_out.view(wide_out.size(0), -1)  # 展平
            # 拼接wide和deep特征
            combined_out = torch.cat([dnn_out, wide_out], dim=-1)
        else:
            combined_out = dnn_out
        
        # 通过共享网络
        shared_out = self.shared_dnn(combined_out)
        
        # 预测ZILN分布参数
        logit_pi = self.pi_head(shared_out).squeeze(-1)  # 零膨胀logit，形状: [batch_size]
        mu = self.mu_head(shared_out).squeeze(-1)        # 均值，形状: [batch_size]
        log_sigma = self.log_sigma_head(shared_out).squeeze(-1)  # 对数标准差，形状: [batch_size]
        
        return torch.stack([logit_pi, mu, log_sigma], dim=1)
    
    #只是预估 ltv
    def predict(self, yhat, bid = None):
        """
        预测期望LTV
        E[LTV] = (1 - pi) * exp(mu + sigma^2/2)
        """
        logit_pi, mu, log_sigma = torch.split(yhat, [1, 1, 1], dim=1)
        pi = torch.sigmoid(logit_pi)
        sigma = torch.exp(log_sigma)
        expected_ltv = (1 - pi) * torch.exp(mu + 0.5 * sigma**2)
        return expected_ltv
    
    def predict_conversion_prob(self, x):
        """
        预测转化概率 P(y > 0) = 1 - pi
        """
        logit_pi, _, _ = self.forward(x)
        pi = torch.sigmoid(logit_pi)
        return 1 - pi
    
    def sample_ltv(self, x, num_samples=1):
        """
        从ZILN分布采样LTV值
        """
        logit_pi, mu, log_sigma = self.forward(x)
        pi = torch.sigmoid(logit_pi)
        sigma = torch.exp(log_sigma)
        
        batch_size = x.size(0)
        
        # 采样是否为零
        is_zero = torch.bernoulli(pi).unsqueeze(-1).expand(-1, num_samples)
        
        # 从对数正态分布采样正值
        log_normal_samples = torch.randn(batch_size, num_samples, device=mu.device) * sigma.unsqueeze(-1) + mu.unsqueeze(-1)
        positive_samples = torch.exp(log_normal_samples)
        
        # 合并零值和正值
        samples = torch.where(is_zero > 0.5, torch.zeros_like(positive_samples), positive_samples)
        return samples


def ziln_expected_ltv_loss(logit_pi, mu, log_sigma, y_true, reduction='mean'):
    """
    基于期望LTV的损失函数（用于回归优化）
    使用均方误差或平均绝对误差
    """
    pi = torch.sigmoid(logit_pi)
    sigma = torch.exp(log_sigma)
    expected_ltv = (1 - pi) * torch.exp(mu + 0.5 * sigma**2)
    
    if reduction == 'mse':
        return F.mse_loss(expected_ltv, y_true)
    elif reduction == 'mae':
        return F.l1_loss(expected_ltv, y_true)
    else:
        return F.mse_loss(expected_ltv, y_true)