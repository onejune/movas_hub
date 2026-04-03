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
====================================

用于处理连续值特征（统计特征、数值特征等），与 EmbeddingSumConcat 处理的稀疏特征互补。

统一接口
--------
所有编码器遵循相同接口：
- __init__(num_features, **kwargs): 只需特征数量，不需要特征名
- output_dim: 输出维度属性
- forward(x): 输入 (batch, num_features)，输出 (batch, output_dim)

支持的编码器
-----------
- linear:   BatchNorm + Linear + Dropout，最常用
- minmax:   Min-Max 归一化到 [0,1]，需要 fit
- standard: Z-score 标准化，需要 fit
- log:      对数变换 log(x+1)，适合长尾分布
- numeric:  每特征独立 MLP (AutoDis 风格)，表达能力最强

使用示例
-------
>>> from dense_feature import create_dense_encoder
>>> encoder = create_dense_encoder('linear', num_features=39, output_dim=32)
>>> dense_out = encoder(dense_tensor)  # [batch, 32]
"""

import torch
import torch.nn as nn
from typing import List


class DenseFeatureLinear(nn.Module):
    """
    线性编码器 (推荐默认选择)
    
    结构: Input -> BatchNorm -> Linear -> Dropout -> Output
    
    适用场景:
    - 特征已经过预处理（如标准化）
    - 需要降维或升维
    - 作为 baseline 编码器
    
    Args:
        num_features: 输入特征数量
        output_dim: 输出维度，None 表示保持原维度
        batch_norm: 是否使用 BatchNorm，默认 True
        dropout: Dropout 比例，默认 0（不使用）
    
    Example:
        >>> encoder = DenseFeatureLinear(39, output_dim=32)
        >>> out = encoder(x)  # [batch, 39] -> [batch, 32]
    """
    
    def __init__(self, num_features: int, output_dim: int = None, 
                 batch_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self._output_dim = output_dim if output_dim else num_features
        
        # BatchNorm 在原始维度上做归一化
        self.bn = nn.BatchNorm1d(num_features) if batch_norm else None
        
        # 线性变换（仅当需要改变维度时）
        self.linear = nn.Linear(num_features, self._output_dim) if output_dim and output_dim != num_features else None
        
        # Dropout 正则化
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        print(f"[DenseFeatureLinear] {num_features} -> {self._output_dim}")
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn:
            # BatchNorm 需要 batch_size > 1，否则跳过
            if x.size(0) > 1:
                x = self.bn(x)
        if self.linear:
            x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DenseFeatureMinMaxScaler(nn.Module):
    """
    Min-Max 归一化
    
    公式: x' = (x - min) / (max - min + eps)
    
    将特征缩放到 [0, 1] 区间，保持原始分布形状。
    
    适用场景:
    - 特征取值范围差异大
    - 需要有界输出
    - 后续使用对尺度敏感的激活函数（如 sigmoid）
    
    注意:
    - 需要先调用 fit() 计算 min/max 统计量
    - 如果不 fit，默认 min=0, max=1（即不变换）
    
    Args:
        num_features: 特征数量
    
    Example:
        >>> encoder = DenseFeatureMinMaxScaler(39)
        >>> encoder.fit(train_data)  # 可选
        >>> out = encoder(x)  # [batch, 39] -> [batch, 39]
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        
        # 使用 register_buffer 保存统计量，会随模型保存/加载
        self.register_buffer("x_min", torch.zeros(num_features))
        self.register_buffer("x_max", torch.ones(num_features))
        self._fitted = False
        
        print(f"[DenseFeatureMinMaxScaler] {num_features} features")
    
    @property
    def output_dim(self) -> int:
        return self.num_features
    
    def fit(self, x: torch.Tensor):
        """从数据计算 min/max 统计量"""
        self.x_min = x.min(dim=0)[0]
        self.x_max = x.max(dim=0)[0]
        self._fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)


class DenseFeatureStandardScaler(nn.Module):
    """
    Z-score 标准化
    
    公式: x' = (x - μ) / σ
    
    将特征转换为均值 0、标准差 1 的分布。
    
    适用场景:
    - 特征近似正态分布
    - 后续使用对尺度敏感的模型（如线性层）
    - 需要消除量纲影响
    
    注意:
    - 需要先调用 fit() 计算 mean/std 统计量
    - 如果不 fit，默认 mean=0, std=1（即不变换）
    
    Args:
        num_features: 特征数量
    
    Example:
        >>> encoder = DenseFeatureStandardScaler(39)
        >>> encoder.fit(train_data)  # 可选
        >>> out = encoder(x)  # [batch, 39] -> [batch, 39]
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
        """从数据计算 mean/std 统计量"""
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        self._fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)


class DenseFeatureLogTransform(nn.Module):
    """
    对数变换
    
    公式: x' = log(x + 1)
    
    压缩大值，拉伸小值，适合处理长尾分布。
    
    适用场景:
    - 特征呈长尾/幂律分布（如曝光数、点击数、金额）
    - 特征取值范围跨越多个数量级
    - 需要压缩极端值的影响
    
    注意:
    - 自动 clamp 负值为 0
    - 无需 fit，无状态变换
    
    Args:
        num_features: 特征数量
    
    Example:
        >>> encoder = DenseFeatureLogTransform(39)
        >>> out = encoder(x)  # [batch, 39] -> [batch, 39]
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        
        print(f"[DenseFeatureLogTransform] {num_features} features")
    
    @property
    def output_dim(self) -> int:
        return self.num_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clamp 防止负值，log1p = log(x+1) 防止 log(0)
        return torch.log1p(torch.clamp(x, min=0))


class DenseFeatureNumericEmbedding(nn.Module):
    """
    Numeric Embedding (AutoDis 风格)
    
    为每个数值特征学习独立的 embedding，类似于稀疏特征的 embedding。
    
    结构: 每个特征 -> Linear(1, hidden) -> ReLU -> Linear(hidden, emb_dim)
    输出: 所有特征的 embedding 拼接，维度 = num_features * embedding_dim
    
    适用场景:
    - 需要学习数值特征的非线性表示
    - 希望数值特征与稀疏 embedding 对齐
    - 特征之间相关性较低，适合独立建模
    
    参考论文:
    - AutoDis: Automatic Discretization for Embedding Numerical Features
    
    Args:
        num_features: 特征数量
        embedding_dim: 每个特征的 embedding 维度，默认 16
        hidden_dim: MLP 隐藏层维度，默认 64
    
    Example:
        >>> encoder = DenseFeatureNumericEmbedding(39, embedding_dim=16)
        >>> out = encoder(x)  # [batch, 39] -> [batch, 39*16=624]
    """
    
    def __init__(self, num_features: int, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # 每个特征一个独立的 MLP
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(num_features)
        ])
        
        print(f"[DenseFeatureNumericEmbedding] {num_features} features x {embedding_dim} dim = {self.output_dim}")
    
    @property
    def output_dim(self) -> int:
        return self.num_features * self.embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_features):
            xi = x[:, i:i+1]  # [batch, 1]
            outputs.append(self.mlps[i](xi))  # [batch, embedding_dim]
        return torch.cat(outputs, dim=-1)  # [batch, num_features * embedding_dim]


# =============================================================================
# 编码器工厂
# =============================================================================

ENCODER_REGISTRY = {
    'linear': DenseFeatureLinear,
    'minmax': DenseFeatureMinMaxScaler,
    'standard': DenseFeatureStandardScaler,
    'log': DenseFeatureLogTransform,
    'numeric': DenseFeatureNumericEmbedding,
}


def create_dense_encoder(encoder_type: str, num_features: int, **kwargs) -> nn.Module:
    """
    创建 dense 特征编码器（工厂函数）
    
    Args:
        encoder_type: 编码器类型
            - 'linear':   BatchNorm + Linear + Dropout (推荐默认)
            - 'minmax':   Min-Max 归一化到 [0,1]
            - 'standard': Z-score 标准化
            - 'log':      对数变换 log(x+1)
            - 'numeric':  每特征独立 MLP (AutoDis)
        num_features: 特征数量
        **kwargs: 编码器特定参数
            - linear:  output_dim, batch_norm, dropout
            - numeric: embedding_dim, hidden_dim
    
    Returns:
        nn.Module: 编码器实例
    
    Raises:
        ValueError: 未知的编码器类型
    
    Example:
        >>> # 基础线性编码器
        >>> encoder = create_dense_encoder('linear', 39, output_dim=32)
        
        >>> # AutoDis 风格
        >>> encoder = create_dense_encoder('numeric', 39, embedding_dim=16)
        
        >>> # 对数变换（无额外参数）
        >>> encoder = create_dense_encoder('log', 39)
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: '{encoder_type}'. "
            f"Supported: {list(ENCODER_REGISTRY.keys())}"
        )
    
    return ENCODER_REGISTRY[encoder_type](num_features, **kwargs)
