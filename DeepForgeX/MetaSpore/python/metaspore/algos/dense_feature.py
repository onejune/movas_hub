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
dense_feature.py - Dense 特征处理模块

支持从 minibatch 中提取连续值特征，并进行标准化、线性变换等处理。

使用方式:
1. 配置文件中指定 dense_features_path: ./conf/dense_features
2. dense_features 文件每行一个特征名
3. 模型中通过 do_extra_work(minibatch) 钩子提取 dense 特征
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class BaseDenseEncoder(nn.Module):
    """Dense特征编码器基类"""
    
    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DenseFeatureLayer(nn.Module):
    """
    Dense 特征处理层
    
    功能:
    - 从配置文件加载 dense 特征名列表
    - 从 minibatch 提取 dense 特征值
    - 支持 BatchNorm 标准化
    - 支持线性变换到指定维度
    - 支持 Dropout
    
    Args:
        dense_features_path: dense 特征配置文件路径，每行一个特征名
        feature_names: 直接指定特征名列表 (与 dense_features_path 二选一)
        output_dim: 输出维度，None 表示保持原始维度
        batch_norm: 是否使用 BatchNorm
        dropout: Dropout 比例
    """
    
    def __init__(self,
                 dense_features_path: str = None,
                 feature_names: List[str] = None,
                 output_dim: int = None,
                 batch_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        
        # 加载特征名列表
        if dense_features_path is not None:
            self.feature_names = self._load_feature_names(dense_features_path)
        elif feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            raise ValueError("Must specify either dense_features_path or feature_names")
        
        self.feature_dim = len(self.feature_names)
        self.output_dim = output_dim if output_dim else self.feature_dim
        
        if self.feature_dim == 0:
            raise ValueError("No dense features specified")
        
        # BatchNorm 层
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.feature_dim)
        else:
            self.bn = None
        
        # 线性变换层 (可选)
        if output_dim and output_dim != self.feature_dim:
            self.linear = nn.Linear(self.feature_dim, output_dim)
        else:
            self.linear = None
        
        # Dropout 层 (可选)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # 存储从 minibatch 提取的数据
        self._dense_output = None
        
        # 特征名到索引的映射 (运行时填充)
        self._feature_indices = None
        
        print(f"[DenseFeatureLayer] Loaded {self.feature_dim} dense features: {self.feature_names[:5]}...")
    
    def _load_feature_names(self, path: str) -> List[str]:
        """从配置文件加载特征名列表"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dense features config not found: {path}")
        
        feature_names = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):  # 跳过空行和注释
                    feature_names.append(name)
        
        return feature_names
    
    def _build_feature_indices(self, column_names: List[str]):
        """构建特征名到列索引的映射"""
        self._feature_indices = []
        for name in self.feature_names:
            if name in column_names:
                idx = column_names.index(name)
                self._feature_indices.append(idx)
            else:
                raise ValueError(f"Dense feature '{name}' not found in minibatch columns: {column_names[:10]}...")
    
    def set_column_names(self, column_names: List[str]):
        """
        设置列名列表，用于构建特征索引映射
        
        在模型初始化后、训练开始前调用一次即可
        """
        self._build_feature_indices(column_names)
    
    def extract(self, minibatch, column_names: List[str] = None):
        """
        从 minibatch 提取 dense 特征
        
        Args:
            minibatch: MiniBatch 对象，包含 tensor 数据
            column_names: 列名列表，用于定位特征位置 (首次调用需要)
        
        在 model.do_extra_work(minibatch) 中调用
        """
        # 首次调用时构建索引映射
        if self._feature_indices is None:
            if column_names is None:
                raise ValueError("column_names required for first extract() call, or call set_column_names() first")
            self._build_feature_indices(column_names)
        
        # 从 minibatch 提取数据
        # minibatch 可能是不同类型，尝试多种方式获取数据
        if hasattr(minibatch, 'tensor'):
            # MiniBatch 对象
            data = minibatch.tensor
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        elif hasattr(minibatch, '__class__') and minibatch.__class__.__name__ == 'DataFrame':
            # Pandas DataFrame - 直接按 dense 特征名提取
            dense_values = minibatch[self.feature_names].values
            self._dense_output = torch.tensor(dense_values, dtype=torch.float32)
        elif isinstance(minibatch, torch.Tensor):
            # 纯 tensor
            data = minibatch
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        elif hasattr(minibatch, 'data'):
            data = minibatch.data
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        elif hasattr(minibatch, 'x'):
            data = minibatch.x
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        else:
            raise ValueError(f"Cannot extract data from minibatch of type {type(minibatch)}")
    
    def forward(self, x=None) -> torch.Tensor:
        """
        返回处理后的 dense 特征
        
        Args:
            x: 占位参数，保持与 EmbeddingSumConcat 接口一致
        
        Returns:
            处理后的 dense 特征 tensor，形状 [batch_size, output_dim]
        """
        if self._dense_output is None:
            raise RuntimeError("Must call extract() before forward()")
        
        out = self._dense_output
        
        # BatchNorm
        if self.bn is not None:
            out = self.bn(out)
        
        # 线性变换
        if self.linear is not None:
            out = self.linear(out)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
    
    def reset(self):
        """重置内部状态，每个 batch 开始前调用"""
        self._dense_output = None
    
    @property
    def feature_count(self) -> int:
        """返回 dense 特征数量"""
        return self.feature_dim


class DenseFeatureEmbedding(nn.Module):
    """
    Dense 特征 Embedding 层
    
    将每个 dense 特征映射到 embedding 空间，然后拼接。
    适用于需要与 sparse embedding 对齐维度的场景。
    
    Args:
        dense_features_path: dense 特征配置文件路径
        feature_names: 直接指定特征名列表
        embedding_dim: 每个特征的 embedding 维度
        batch_norm: 是否使用 BatchNorm
    """
    
    def __init__(self,
                 dense_features_path: str = None,
                 feature_names: List[str] = None,
                 embedding_dim: int = 8,
                 batch_norm: bool = True):
        super().__init__()
        
        # 加载特征名
        if dense_features_path is not None:
            self.feature_names = self._load_feature_names(dense_features_path)
        elif feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            raise ValueError("Must specify either dense_features_path or feature_names")
        
        self.feature_dim = len(self.feature_names)
        self.embedding_dim = embedding_dim
        self.output_dim = self.feature_dim * embedding_dim
        
        # 每个 dense 特征一个线性层，映射到 embedding_dim
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(self.feature_dim)
        ])
        
        # BatchNorm (在 embedding 之前)
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.feature_dim)
        else:
            self.bn = None
        
        self._dense_output = None
        self._feature_indices = None
        
        print(f"[DenseFeatureEmbedding] {self.feature_dim} features x {embedding_dim} dim = {self.output_dim}")


class DenseFeatureMinMaxScaler(BaseDenseEncoder):
    """
    MinMaxScaler：将特征缩放到 [0, 1] 区间
    
    x' = (x - min) / (max - min + eps)
    """
    
    def __init__(self, 
                 dense_features: List[str],
                 fit_on_data: bool = True):
        super().__init__()
        
        self.feature_names = dense_features
        self.feature_dim = len(self.feature_names)
        self.fit_on_data = fit_on_data
        
        # 统计参数
        self.register_buffer("x_min", torch.zeros(self.feature_dim))
        self.register_buffer("x_max", torch.ones(self.feature_dim))
        self._fitted = False
        
        print(f"[DenseFeatureMinMaxScaler] Initialized for {self.feature_dim} features")
    
    def fit(self, x: torch.Tensor):
        """用数据拟合缩放参数"""
        self.x_min = torch.min(x, dim=0)[0]
        self.x_max = torch.max(x, dim=0)[0]
        self._fitted = True
        print(f"[DenseFeatureMinMaxScaler] Fit complete: min={self.x_min[:3]}, max={self.x_max[:3]}")
    
    def fit_from_numpy(self, x: np.ndarray):
        """用 numpy 数组拟合缩放参数"""
        x_tensor = torch.from_numpy(x).float()
        self.fit(x_tensor)
    
    @property
    def output_dim(self) -> int:
        return self.feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted and self.fit_on_data:
            # 首次调用时自动拟合
            self.fit(x)
        
        x_min = self.x_min.to(x.device)
        x_max = self.x_max.to(x.device)
        return (x - x_min) / (x_max - x_min + 1e-8)


class DenseFeatureStandardScaler(BaseDenseEncoder):
    """
    StandardScaler：Z-score 标准化，均值0方差1
    
    x' = (x - μ) / σ
    """
    
    def __init__(self, 
                 dense_features: List[str],
                 fit_on_data: bool = True):
        super().__init__()
        
        self.feature_names = dense_features
        self.feature_dim = len(self.feature_names)
        self.fit_on_data = fit_on_data
        
        # 统计参数
        self.register_buffer("mean", torch.zeros(self.feature_dim))
        self.register_buffer("std", torch.ones(self.feature_dim))
        self._fitted = False
        
        print(f"[DenseFeatureStandardScaler] Initialized for {self.feature_dim} features")
    
    def fit(self, x: torch.Tensor):
        """用数据拟合标准化参数"""
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0, unbiased=False)  # 使用总体标准差
        self._fitted = True
        print(f"[DenseFeatureStandardScaler] Fit complete: mean={self.mean[:3]}, std={self.std[:3]}")
    
    def fit_from_numpy(self, x: np.ndarray):
        """用 numpy 数组拟合标准化参数"""
        x_tensor = torch.from_numpy(x).float()
        self.fit(x_tensor)
    
    @property
    def output_dim(self) -> int:
        return self.feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted and self.fit_on_data:
            # 首次调用时自动拟合
            self.fit(x)
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / (std + 1e-8)


class DenseFeatureNumericEmbedding(BaseDenseEncoder):
    """
    NumericEmbedding：每个特征独立的小 MLP 映射
    
    每个标量特征通过独立 MLP 映射到 embedding 空间
    适用于需要非线性变换的连续特征
    """
    
    def __init__(self,
                 dense_features: List[str],
                 embedding_dim: int = 16,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.feature_names = dense_features
        self.feature_dim = len(self.feature_names)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 为每个特征创建独立的 MLP
        # 每个特征: Linear(1->hidden) -> ReLU -> Linear(hidden->emb_dim)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(self.feature_dim)
        ])
        
        print(f"[DenseFeatureNumericEmbedding] {self.feature_dim} features -> {embedding_dim} dim each")
    
    @property
    def output_dim(self) -> int:
        return self.feature_dim * self.embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        outputs = []
        for i in range(self.feature_dim):
            xi = x[:, i:i+1]  # (batch, 1)
            emb = self.mlps[i](xi)  # (batch, emb_dim)
            outputs.append(emb)
        return torch.cat(outputs, dim=-1)  # (batch, n_features * emb_dim)


class DenseFeatureLogTransform(BaseDenseEncoder):
    """
    LogTransform：对数变换，适用于长尾分布特征
    
    x' = log(x + 1)  # 使用 log1p 避免精度问题
    """
    
    def __init__(self, 
                 dense_features: List[str]):
        super().__init__()
        
        self.feature_names = dense_features
        self.feature_dim = len(self.feature_names)
        
        print(f"[DenseFeatureLogTransform] Initialized for {self.feature_dim} features")
    
    @property
    def output_dim(self) -> int:
        return self.feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先确保非负，然后应用 log1p
        x_clamped = torch.clamp(x, min=0.0)
        return torch.log1p(x_clamped)
    
    def _load_feature_names(self, path: str) -> List[str]:
        """从配置文件加载特征名列表"""
        feature_names = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    feature_names.append(name)
        return feature_names
    
    def _build_feature_indices(self, column_names: List[str]):
        """构建特征名到列索引的映射"""
        self._feature_indices = []
        for name in self.feature_names:
            if name in column_names:
                idx = column_names.index(name)
                self._feature_indices.append(idx)
            else:
                raise ValueError(f"Dense feature '{name}' not found in columns")
    
    def extract(self, minibatch, column_names: List[str] = None):
        """从 minibatch 提取 dense 特征"""
        if self._feature_indices is None:
            if column_names is None:
                raise ValueError("column_names required for first extract() call")
            self._build_feature_indices(column_names)
        
        # 从 minibatch 提取数据
        # minibatch 可能是不同类型，尝试多种方式获取数据
        if hasattr(minibatch, 'tensor'):
            # MiniBatch 对象
            data = minibatch.tensor
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        elif hasattr(minibatch, '__class__') and minibatch.__class__.__name__ == 'DataFrame':
            # Pandas DataFrame - 直接按 dense 特征名提取
            dense_values = minibatch[self.feature_names].values
            self._dense_output = torch.tensor(dense_values, dtype=torch.float32)
        elif hasattr(minibatch, 'data'):
            data = minibatch.data
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        elif hasattr(minibatch, 'x'):
            data = minibatch.x
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        elif isinstance(minibatch, torch.Tensor):
            # 纯 tensor
            data = minibatch
            dense_cols = [data[:, idx:idx+1] for idx in self._feature_indices]
            self._dense_output = torch.cat(dense_cols, dim=1).float()
        else:
            raise ValueError(f"Cannot extract data from minibatch of type {type(minibatch)}")
    
    def forward(self, x=None) -> torch.Tensor:
        """返回 dense 特征的 embedding 拼接"""
        if self._dense_output is None:
            raise RuntimeError("Must call extract() before forward()")
        
        out = self._dense_output
        
        # BatchNorm
        if self.bn is not None:
            out = self.bn(out)
        
        # 每个特征单独 embedding 然后拼接
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            feat_emb = emb_layer(out[:, i:i+1])  # [batch, embedding_dim]
            emb_list.append(feat_emb)
        
        return torch.cat(emb_list, dim=1)  # [batch, feature_dim * embedding_dim]
    
    def reset(self):
        """重置内部状态"""
        self._dense_output = None
    
    @property
    def feature_count(self) -> int:
        """返回 dense 特征数量"""
        return self.feature_dim
