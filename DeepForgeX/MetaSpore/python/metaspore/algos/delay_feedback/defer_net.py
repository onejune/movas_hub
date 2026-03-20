"""
DEFER DNN 模型 - 多窗口延迟反馈建模

输出 4 个 logits: cv, time15, time30, time60

Reference: DEFER (Delayed Feedback Modeling with Adaptive Window Selection)
"""
import torch
import torch.nn as nn
import metaspore as ms
from ..layers import MLPLayer


class DeferDNN(nn.Module):
    """
    DEFER DNN 模型
    
    结构:
        Embedding -> SharedDNN -> 4个输出头 (cv, t15, t30, t60)
    
    Args:
        embedding_dim: Embedding 维度
        column_name_path: 特征列名文件路径
        combine_schema_path: 特征组合配置路径
        dnn_hidden_units: DNN 隐层单元数
        net_dropout: Dropout 率
        batch_norm: 是否使用 BatchNorm
        ftrl_*: FTRL 优化器参数
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
        
        # Embedding 层
        self.sparse_embedding = ms.EmbeddingSumConcat(
            embedding_dim, column_name_path, combine_schema_path
        )
        self.sparse_embedding.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
        )
        self.sparse_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        input_dim = self.sparse_embedding.feature_count * embedding_dim
        
        # 共享 DNN 层 (去掉最后一层，作为共享特征提取)
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
        
        # 4 个输出头
        self.cv_head = MLPLayer(
            input_dim=hidden_dim, output_dim=1, 
            hidden_units=[last_hidden],
            hidden_activations=dnn_hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
        )
        self.time15_head = MLPLayer(
            input_dim=hidden_dim, output_dim=1,
            hidden_units=[last_hidden],
            hidden_activations=dnn_hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
        )
        self.time30_head = MLPLayer(
            input_dim=hidden_dim, output_dim=1,
            hidden_units=[last_hidden],
            hidden_activations=dnn_hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
        )
        self.time60_head = MLPLayer(
            input_dim=hidden_dim, output_dim=1,
            hidden_units=[last_hidden],
            hidden_activations=dnn_hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
        )
        
        self.sigmoid = nn.Sigmoid()
        
        print(f"DeferDNN 初始化: input_dim={input_dim}, shared_units={shared_units}, "
              f"hidden_dim={hidden_dim}, last_hidden={last_hidden}")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (Spark DataFrame batch)
        
        Returns:
            logits: [B, 4] - (cv, t15, t30, t60)
        """
        emb = self.sparse_embedding(x)
        h = self.shared_dnn(emb)
        
        cv_logits = self.cv_head(h)
        time15_logits = self.time15_head(h)
        time30_logits = self.time30_head(h)
        time60_logits = self.time60_head(h)
        
        # 拼接成 [B, 4]
        logits = torch.cat([cv_logits, time15_logits, time30_logits, time60_logits], dim=1)
        return logits
    
    def predict(self, logits, minibatch=None):
        """
        预测转化概率
        
        使用 60min 窗口: P(cv) * P(t<=60|cv)
        
        Args:
            logits: [B, 4] 模型输出
            minibatch: 可选，用于获取预测窗口配置
        
        Returns:
            pred: [B, 1] 转化概率
        """
        cv_prob = self.sigmoid(logits[:, 0:1])
        time60_prob = self.sigmoid(logits[:, 3:4])
        return cv_prob * time60_prob
