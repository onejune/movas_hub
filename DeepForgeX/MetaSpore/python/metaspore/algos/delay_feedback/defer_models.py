#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEFER WinAdapt 模型 - 使用 MetaSpore 接口

参考 widedeep_net.py，使用 ms.EmbeddingSumConcat 处理稀疏特征

修正版本:
1. forward 返回 Tensor (logits)，不是 dict
2. 与 MetaSpore PyTorchEstimator 兼容
"""
import torch
import torch.nn as nn
import metaspore as ms

from ..layers import MLPLayer


class WinAdaptDNN(torch.nn.Module):
    """
    WinAdapt 4 输出头模型
    
    输出:
    - logits: (batch_size, 4) - [cv_logit, time_w1_logit, time_w2_logit, time_w3_logit]
    
    最终预测: window_prob = sigmoid(cv_logit) * sigmoid(time_wx_logit)
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
        
        # 稀疏特征 Embedding (使用 MetaSpore 接口)
        self.sparse_embedding = ms.EmbeddingSumConcat(
            embedding_dim,
            column_name_path,
            combine_schema_path
        )
        self.sparse_embedding.updater = ms.AdamTensorUpdater(adam_learning_rate)
        self.sparse_embedding.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # 计算 DNN 输入维度
        dnn_input_dim = self.sparse_embedding.feature_count * embedding_dim
        
        # 共享 DNN 层
        self.shared_dnn = MLPLayer(
            input_dim=dnn_input_dim,
            output_dim=dnn_hidden_units[-1],  # 输出到最后一层隐藏层
            hidden_units=dnn_hidden_units[:-1],  # 除了最后一层
            hidden_activations=dnn_hidden_activations,
            final_activation=dnn_hidden_activations,  # 最后也用激活
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=True
        )
        
        # 4 个输出头 (合并成一个 Linear，输出 4 维)
        hidden_dim = dnn_hidden_units[-1]
        self.output_layer = nn.Linear(hidden_dim, 4)
        
        # 初始化
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 稀疏特征输入 (batch_size, num_features)
        
        Returns:
            logits: (batch_size, 4) - [cv_logit, time_w1_logit, time_w2_logit, time_w3_logit]
        """
        # Embedding
        embed = self.sparse_embedding(x)  # (batch, feature_count * embed_dim)
        
        # 共享 DNN
        hidden = self.shared_dnn(embed)  # (batch, hidden_dim)
        
        # 输出层: 4 维 logits
        logits = self.output_layer(hidden)  # (batch, 4)
        
        return logits

    def predict(self, logits, minibatch=None):
        """
        返回 cv 预测概率用于 AUC 计算
        
        Args:
            logits: (batch_size, 4) - forward 的输出
            minibatch: 可选，未使用
        
        Returns:
            cv_prob: (batch_size,) - cv 转化概率
        """
        cv_logit = logits[:, 0]  # 第一列是 cv_logit
        return torch.sigmoid(cv_logit)  # (batch,)


# 测试
if __name__ == "__main__":
    print("WinAdaptDNN 模型测试")
    print("注意: 需要在 MetaSpore 环境中测试，需要 combine_schema 文件")
