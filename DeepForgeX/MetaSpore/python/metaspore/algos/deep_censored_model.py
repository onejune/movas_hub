
import torch
import metaspore as ms
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from .layers import MLPLayer

class DeepCensoredModel(torch.nn.Module):
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
        
        # Wide部分：线性特征
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim,
                                                wide_column_name_path,
                                                wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # Deep部分：嵌入层
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim,
                                           deep_column_name_path,
                                           deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # Deep部分：共享主干网络
        self.shared_dnn = MLPLayer(input_dim = self.dnn_sparse.feature_count * deep_embedding_dim,
                                output_dim = dnn_hidden_units[-1],  # 输出最后一层维度
                                hidden_units = dnn_hidden_units[:-1],  # 除最后一层外的所有层
                                hidden_activations = dnn_hidden_activations,
                                final_activation = dnn_hidden_activations,
                                dropout_rates = net_dropout,
                                batch_norm = batch_norm,
                                use_bias = use_bias,
                                input_norm = True)
        
        # 分布参数输出头：均值μ
        self.mu_head = MLPLayer(input_dim = dnn_hidden_units[-1],
                               output_dim = 1,
                               hidden_units = [16],  # 小型MLP
                               hidden_activations = dnn_hidden_activations,
                               final_activation = None,  # 线性输出
                               dropout_rates = net_dropout,
                               batch_norm = batch_norm,
                               use_bias = use_bias)
        
        # 分布参数输出头：对数标准差log(σ)
        self.log_sigma_head = MLPLayer(input_dim = dnn_hidden_units[-1],
                                      output_dim = 1,
                                      hidden_units = [16],  # 小型MLP
                                      hidden_activations = dnn_hidden_activations,
                                      final_activation = None,  # 线性输出
                                      dropout_rates = net_dropout,
                                      batch_norm = batch_norm,
                                      use_bias = use_bias)
        self.final_activation = torch.nn.Sigmoid()
        # 默认分布参数（用于初始化和边界情况）
        #self.register_buffer('default_mu', torch.tensor(0.0))
        #self.register_buffer('default_log_sigma', torch.tensor(0.0))

    def forward(self, x):
        """
        前向传播，输出分布参数
        返回: (mu, log_sigma)
        """
        # 获取deep特征嵌入
        dnn_out = self.dnn_sparse(x)
        dnn_out = self.shared_dnn(dnn_out)
        
        # Wide部分（如果启用）
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
            # 将wide部分加到共享网络输出上
            dnn_out = dnn_out + wide_out.expand_as(dnn_out)
        
        # 预测分布参数
        mu = self.mu_head(dnn_out).squeeze(-1)  # 均值，形状: [batch_size]
        log_sigma = self.log_sigma_head(dnn_out).squeeze(-1)  # 对数标准差，形状: [batch_size]
        
        return torch.stack([mu, log_sigma], dim=1)
    
    def predict(self, yhat, bid = None):
        """
        预测给定出价下的竞胜率
        Args:
            x: 输入特征
            bid: 出价（可以是标量或与x同batch_size的tensor）
        Returns:
            win_rate: 竞胜率，形状: [batch_size]
        """
        mu, log_sigma = torch.split(yhat, [1, 1], dim=1)
        sigma = torch.exp(log_sigma)
        
        # 计算 P(log(v) < log(bid)) = P(v < bid)，假设v服从对数正态分布
        # 即 log(v) ~ N(mu, sigma^2)
        log_bid = torch.log(bid + 1e-8)  # 避免除零
        z_score = (log_bid - mu) / sigma
        
        # 使用标准正态CDF计算竞胜率
        win_rate = torch.erf(z_score / np.sqrt(2)) / 2 + 0.5  # 等价于标准正态CDF
        
        return win_rate
    
    def sample_price(self, x, num_samples=1):
        """
        从预测的价格分布中采样
        Args:
            x: 输入特征
            num_samples: 采样数量
        Returns:
            samples: 价格样本，形状: [batch_size, num_samples]
        """
        mu, log_sigma = self.forward(x)
        sigma = torch.exp(log_sigma)
        
        # 从正态分布采样 log(price)
        log_price_samples = torch.randn(x.size(0), num_samples, device=mu.device) * sigma.unsqueeze(-1) + mu.unsqueeze(-1)
        
        # 转换回价格空间
        price_samples = torch.exp(log_price_samples)
        return price_samples