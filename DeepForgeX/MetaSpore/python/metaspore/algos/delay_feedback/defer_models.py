#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEFER 延迟反馈模型

包含两种网络结构，均输出 4 维 logits，可共用 delay_win_select_loss:

1. WinAdaptDNN (推荐)
   - 结构: Embedding → SharedDNN → Linear(hidden, 4)
   - 特点: 参数少，单个输出层，适合长时间窗口 (24/48/72h)
   - 用途: defer_v2

2. DeferDNN
   - 结构: Embedding → SharedDNN → 4 个独立 MLPLayer 输出头
   - 特点: 参数多，每个窗口独立建模，适合短时间窗口 (15/30/60min)
   - 用途: defer_v1

两者输出格式一致: (batch, 4) = [cv_logit, time_w1, time_w2, time_w3]
损失函数: delay_win_select_loss (14 维标签)

Reference:
- DEFER: Delayed Feedback Modeling for Conversion Rate Prediction
- WinAdapt: Window-Adaptive Delayed Feedback Modeling
"""
import torch
import torch.nn as nn
import metaspore as ms

from ..layers import MLPLayer


# =============================================================================
# WinAdaptDNN - 单输出层版本 (推荐)
# =============================================================================

class WinAdaptDNN(nn.Module):
    """
    WinAdapt 延迟反馈模型 - 单输出层版本
    
    网络结构:
        Sparse Features → EmbeddingSumConcat → SharedDNN → Linear(4)
    
    输出:
        logits: (batch, 4) = [cv_logit, time_w1, time_w2, time_w3]
        
        - cv_logit: 转化概率 logit
        - time_w1/w2/w3: 各时间窗口内转化的条件概率 logit
        
    预测:
        P(转化在窗口w内) = sigmoid(cv_logit) × sigmoid(time_w_logit)
    
    适用场景:
        - 长时间窗口 (小时/天级别): 24h, 48h, 72h
        - 参数量较少，训练更稳定
    
    Args:
        embedding_dim: Embedding 维度 (默认 8)
        column_name_path: 特征列名文件
        combine_schema_path: 特征组合配置
        dnn_hidden_units: DNN 隐层单元 (默认 [256, 256, 128])
        dnn_hidden_activations: 激活函数 (默认 ReLU)
        net_dropout: Dropout 率 (默认 0)
        batch_norm: 是否使用 BatchNorm (默认 False)
        adam_learning_rate: Embedding Adam 学习率 (默认 0.001)
        ftrl_*: FTRL 优化器参数 (未使用，保留兼容)
    """
    
    def __init__(self,
                 embedding_dim=8,
                 column_name_path=None,
                 combine_schema_path=None,
                 dnn_hidden_units=[256, 256, 128],
                 dnn_hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 use_bias=True,
                 adam_learning_rate=0.001,
                 ftrl_l1=1.0,
                 ftrl_l2=10.0,
                 ftrl_alpha=0.005,
                 ftrl_beta=0.05,
                 **kwargs):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Sparse Embedding (MetaSpore 接口)
        self.sparse_embedding = ms.EmbeddingSumConcat(
            embedding_dim,
            column_name_path,
            combine_schema_path
        )
        self.sparse_embedding.updater = ms.AdamTensorUpdater(adam_learning_rate)
        self.sparse_embedding.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # DNN 输入维度
        dnn_input_dim = self.sparse_embedding.feature_count * embedding_dim
        
        # 共享 DNN (输出到最后一层隐藏维度)
        self.shared_dnn = MLPLayer(
            input_dim=dnn_input_dim,
            output_dim=dnn_hidden_units[-1],
            hidden_units=dnn_hidden_units[:-1],
            hidden_activations=dnn_hidden_activations,
            final_activation=dnn_hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=True
        )
        
        # 输出层: 4 维 logits
        hidden_dim = dnn_hidden_units[-1]
        self.output_layer = nn.Linear(hidden_dim, 4)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
        print(f"[WinAdaptDNN] input_dim={dnn_input_dim}, hidden={dnn_hidden_units}, output=4")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: Sparse 特征输入
        
        Returns:
            logits: (batch, 4) - [cv, w1, w2, w3]
        """
        embed = self.sparse_embedding(x)
        hidden = self.shared_dnn(embed)
        logits = self.output_layer(hidden)
        return logits

    def predict(self, logits, minibatch=None):
        """
        预测转化概率 (用于 AUC 评估)
        
        返回 cv 概率，不含时间窗口因子
        
        Args:
            logits: (batch, 4) - forward 输出
        
        Returns:
            cv_prob: (batch,) - 转化概率
        """
        cv_logit = logits[:, 0]
        return torch.sigmoid(cv_logit)


# =============================================================================
# DeferDNN - 多输出头版本
# =============================================================================

class DeferDNN(nn.Module):
    """
    DEFER 延迟反馈模型 - 多输出头版本
    
    网络结构:
        Sparse Features → EmbeddingSumConcat → SharedDNN → 4 × MLPLayer
        
        每个输出头独立建模:
        - cv_head: 转化概率
        - time_w1_head: 窗口1 条件概率
        - time_w2_head: 窗口2 条件概率  
        - time_w3_head: 窗口3 条件概率
    
    输出:
        logits: (batch, 4) = [cv_logit, time_w1, time_w2, time_w3]
    
    预测:
        P(转化在窗口w内) = sigmoid(cv_logit) × sigmoid(time_w_logit)
        默认使用最大窗口: P = sigmoid(cv) × sigmoid(w3)
    
    适用场景:
        - 短时间窗口 (分钟级别): 15min, 30min, 60min
        - 各窗口差异大，需要独立建模
    
    Args:
        embedding_dim: Embedding 维度 (默认 16)
        column_name_path: 特征列名文件
        combine_schema_path: 特征组合配置
        dnn_hidden_units: DNN 隐层单元 (默认 [512, 256, 128])
        net_dropout: Dropout 率 (默认 0.3)
        batch_norm: 是否使用 BatchNorm (默认 True)
        ftrl_*: FTRL 优化器参数 (Embedding 使用 FTRL)
    """
    
    def __init__(self,
                 embedding_dim=16,
                 column_name_path=None,
                 combine_schema_path=None,
                 dnn_hidden_units=[512, 256, 128],
                 dnn_hidden_activations="ReLU",
                 net_dropout=0.3,
                 batch_norm=True,
                 use_bias=True,
                 sparse_init_var=0.01,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 **kwargs):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Sparse Embedding (FTRL 优化器)
        self.sparse_embedding = ms.EmbeddingSumConcat(
            embedding_dim, column_name_path, combine_schema_path
        )
        self.sparse_embedding.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
        )
        self.sparse_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        input_dim = self.sparse_embedding.feature_count * embedding_dim
        
        # 共享 DNN (去掉最后一层作为共享特征)
        if len(dnn_hidden_units) > 1:
            shared_units = dnn_hidden_units[:-1]
            hidden_dim = dnn_hidden_units[-2]
        else:
            shared_units = []
            hidden_dim = input_dim
        
        if shared_units:
            self.shared_dnn = MLPLayer(
                input_dim=input_dim,
                output_dim=None,
                hidden_units=shared_units,
                hidden_activations=dnn_hidden_activations,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
                use_bias=use_bias,
            )
        else:
            self.shared_dnn = nn.Identity()
            hidden_dim = input_dim
        
        last_hidden = dnn_hidden_units[-1] if dnn_hidden_units else 64
        
        # 4 个独立输出头
        head_kwargs = dict(
            input_dim=hidden_dim,
            output_dim=1,
            hidden_units=[last_hidden],
            hidden_activations=dnn_hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
        )
        self.cv_head = MLPLayer(**head_kwargs)
        self.time_w1_head = MLPLayer(**head_kwargs)
        self.time_w2_head = MLPLayer(**head_kwargs)
        self.time_w3_head = MLPLayer(**head_kwargs)
        
        print(f"[DeferDNN] input_dim={input_dim}, shared={shared_units}, "
              f"hidden_dim={hidden_dim}, head_hidden={last_hidden}")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: Sparse 特征输入
        
        Returns:
            logits: (batch, 4) - [cv, w1, w2, w3]
        """
        emb = self.sparse_embedding(x)
        h = self.shared_dnn(emb)
        
        cv = self.cv_head(h)
        w1 = self.time_w1_head(h)
        w2 = self.time_w2_head(h)
        w3 = self.time_w3_head(h)
        
        logits = torch.cat([cv, w1, w2, w3], dim=1)
        return logits
    
    def predict(self, logits, minibatch=None):
        """
        预测转化概率 (用于 AUC 评估)
        
        使用最大窗口联合概率: P(cv) × P(t ≤ w3 | cv)
        
        Args:
            logits: (batch, 4) - forward 输出
        
        Returns:
            prob: (batch,) - 联合转化概率
        """
        cv_prob = torch.sigmoid(logits[:, 0])
        w3_prob = torch.sigmoid(logits[:, 3])
        return cv_prob * w3_prob


# =============================================================================
# 工厂函数
# =============================================================================

def create_defer_model(model_type: str, **kwargs) -> nn.Module:
    """
    创建 DEFER 模型
    
    Args:
        model_type: 模型类型
            - "WinAdaptDNN": 单输出层版本 (推荐，用于长窗口)
            - "DeferDNN": 多输出头版本 (用于短窗口)
        **kwargs: 模型参数
    
    Returns:
        model: nn.Module
    
    Example:
        model = create_defer_model("WinAdaptDNN", embedding_dim=8, ...)
    """
    model_type = model_type.lower()
    
    if model_type in ("winadaptdnn", "winadapt", "v2"):
        return WinAdaptDNN(**kwargs)
    elif model_type in ("deferdnn", "defer", "v1"):
        return DeferDNN(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Choose from: WinAdaptDNN, DeferDNN")


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEFER 模型测试")
    print("注意: 需要在 MetaSpore 环境中测试 (需要 combine_schema 文件)")
    print("=" * 60)
    
    print("\n可用模型:")
    print("  - WinAdaptDNN: 单输出层，适合长窗口 (24/48/72h)")
    print("  - DeferDNN: 多输出头，适合短窗口 (15/30/60min)")
    print("\n两者输出格式一致: (batch, 4) = [cv, w1, w2, w3]")
    print("共用损失函数: delay_win_select_loss")
