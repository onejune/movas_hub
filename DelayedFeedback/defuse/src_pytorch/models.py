"""
DEFUSE 模型定义

模型架构:
- Embedding 层: 每个类别特征一个 Embedding
- 共享 MLP: 提取特征表示
- 三个输出头:
  - cv_head: CVR 预测
  - tn_head: True Negative 分类
  - dp_head: Delayed Positive 分类
"""

from typing import List, Dict

import torch
import torch.nn as nn


class DEFUSEModel(nn.Module):
    """
    DEFUSE 模型
    
    三头架构: cv (CVR), tn (True Negative), dp (Delayed Positive)
    """
    
    def __init__(self, 
                 vocab_sizes: List[int],
                 embed_dim: int = 8,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1):
        """
        Args:
            vocab_sizes: 每个特征的词表大小
            embed_dim: Embedding 维度
            hidden_dims: MLP 隐藏层维度
            dropout: Dropout 率
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [128, 64]
        
        # Embedding 层
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # 共享 MLP
        input_dim = len(vocab_sizes) * embed_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # 三个输出头
        self.cv_head = nn.Linear(prev_dim, 1)  # CVR 预测
        self.tn_head = nn.Linear(prev_dim, 1)  # True Negative
        self.dp_head = nn.Linear(prev_dim, 1)  # Delayed Positive
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight[1:])  # 跳过 padding
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch_size, num_features] 特征索引
        
        Returns:
            字典包含:
            - cv_logits: [batch_size] CVR logits
            - tn_logits: [batch_size] TN logits
            - dp_logits: [batch_size] DP logits
        """
        # Embedding
        embeds = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        h = torch.cat(embeds, dim=-1)  # [batch, num_features * embed_dim]
        
        # 共享 MLP
        h = self.shared(h)
        
        # 三个头
        return {
            'cv_logits': self.cv_head(h).squeeze(-1),
            'tn_logits': self.tn_head(h).squeeze(-1),
            'dp_logits': self.dp_head(h).squeeze(-1),
        }
    
    def get_num_params(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def freeze_tn_dp(self):
        """冻结 tn/dp 头 (用于 Stage 2)"""
        for param in self.tn_head.parameters():
            param.requires_grad = False
        for param in self.dp_head.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
