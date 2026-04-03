#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
dense_feature.py - Dense 特征编码器

所有编码器统一接口：
- __init__(num_features, **kwargs): 只需要特征数量
- output_dim: 输出维度
- forward(x): x 是 (batch, num_features) 的 tensor
"""

import torch
import torch.nn as nn
from typing import List


class DenseFeatureLinear(nn.Module):
    """
    线性编码器：BatchNorm + Linear + Dropout
    """
    
    def __init__(self, num_features: int, output_dim: int = None, 
                 batch_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self._output_dim = output_dim if output_dim else num_features
        
        self.bn = nn.BatchNorm1d(num_features) if batch_norm else None
        self.linear = nn.Linear(num_features, self._output_dim) if output_dim and output_dim != num_features else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        print(f"[DenseFeatureLinear] {num_features} -> {self._output_dim}")
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn:
            x = self.bn(x)
        if self.linear:
            x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DenseFeatureMinMaxScaler(nn.Module):
    """
    MinMax 归一化：x' = (x - min) / (max - min + eps)
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.register_buffer("x_min", torch.zeros(num_features))
        self.register_buffer("x_max", torch.ones(num_features))
        self._fitted = False
        
        print(f"[DenseFeatureMinMaxScaler] {num_features} features")
    
    @property
    def output_dim(self) -> int:
        return self.num_features
    
    def fit(self, x: torch.Tensor):
        self.x_min = x.min(dim=0)[0]
        self.x_max = x.max(dim=0)[0]
        self._fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)


class DenseFeatureStandardScaler(nn.Module):
    """
    Z-score 标准化：x' = (x - μ) / σ
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))
        self._fitted = False
        
        print(f"[DenseFeatureStandardScaler] {num_features} features")
    
    @property
    def output_dim(self) -> int:
        return self.num_features
    
    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        self._fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)


class DenseFeatureLogTransform(nn.Module):
    """
    对数变换：x' = log(x + 1)
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        
        print(f"[DenseFeatureLogTransform] {num_features} features")
    
    @property
    def output_dim(self) -> int:
        return self.num_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(x, min=0))


class DenseFeatureNumericEmbedding(nn.Module):
    """
    NumericEmbedding (AutoDis)：每个特征独立 MLP
    
    每个特征: Linear(1->hidden) -> ReLU -> Linear(hidden->emb_dim)
    输出维度: num_features * embedding_dim
    """
    
    def __init__(self, num_features: int, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(num_features)
        ])
        
        print(f"[DenseFeatureNumericEmbedding] {num_features} features x {embedding_dim} dim")
    
    @property
    def output_dim(self) -> int:
        return self.num_features * self.embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_features):
            xi = x[:, i:i+1]
            outputs.append(self.mlps[i](xi))
        return torch.cat(outputs, dim=-1)


# 编码器工厂
ENCODER_REGISTRY = {
    'linear': DenseFeatureLinear,
    'minmax': DenseFeatureMinMaxScaler,
    'standard': DenseFeatureStandardScaler,
    'log': DenseFeatureLogTransform,
    'numeric': DenseFeatureNumericEmbedding,
}


def create_dense_encoder(encoder_type: str, num_features: int, **kwargs) -> nn.Module:
    """
    创建 dense 特征编码器
    
    Args:
        encoder_type: linear, minmax, standard, log, numeric
        num_features: 特征数量
        **kwargs: 编码器特定参数
    
    Returns:
        编码器实例
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {encoder_type}. Supported: {list(ENCODER_REGISTRY.keys())}")
    
    return ENCODER_REGISTRY[encoder_type](num_features, **kwargs)
