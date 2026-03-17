# -*- coding: utf-8 -*-
"""
特征配置模块
============
支持从 YAML 文件加载特征配置，方便修改和版本管理。
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import yaml


@dataclass
class FeatureConfig:
    """特征配置类
    
    Example:
        >>> # 从 YAML 文件加载
        >>> config = FeatureConfig.from_yaml('features.yaml')
        
        >>> # 或者直接创建
        >>> config = FeatureConfig()
        >>> config.add_sparse_feature('new_feature')
        >>> config.add_cross_feature('country', 'bundle')
        >>> print(config.all_features)
    """
    
    # 稀疏特征（类别型）
    sparse_features: List[str] = field(default_factory=list)
    
    # 稠密特征（数值型）
    dense_features: List[str] = field(default_factory=list)
    
    # 交叉特征 (col1, col2) -> col1_x_col2
    cross_features: List[Tuple[str, str]] = field(default_factory=list)
    
    # 排除的特征
    exclude_features: List[str] = field(default_factory=lambda: [
        'req_hour',
        'sample_bid_date',
        'diff_hours',
    ])
    
    # 标签列
    label_col: str = 'label'
    
    # 高基数特征阈值（出现次数少于此值的归为 OTHER）
    high_cardinality_threshold: int = 50
    
    # Embedding 维度
    embed_dim: int = 16
    
    def add_sparse_feature(self, name: str) -> 'FeatureConfig':
        """添加稀疏特征"""
        if name not in self.sparse_features:
            self.sparse_features.append(name)
        return self
    
    def add_dense_feature(self, name: str) -> 'FeatureConfig':
        """添加稠密特征"""
        if name not in self.dense_features:
            self.dense_features.append(name)
        return self
    
    def add_cross_feature(self, col1: str, col2: str) -> 'FeatureConfig':
        """添加交叉特征"""
        pair = (col1, col2)
        if pair not in self.cross_features:
            self.cross_features.append(pair)
        return self
    
    def remove_feature(self, name: str) -> 'FeatureConfig':
        """移除特征"""
        if name in self.sparse_features:
            self.sparse_features.remove(name)
        if name in self.dense_features:
            self.dense_features.remove(name)
        return self
    
    @property
    def cross_feature_names(self) -> List[str]:
        """生成交叉特征名"""
        return [f'{c1}_x_{c2}' for c1, c2 in self.cross_features]
    
    @property
    def all_sparse_features(self) -> List[str]:
        """所有稀疏特征（含交叉）"""
        return self.sparse_features + self.cross_feature_names
    
    @property
    def all_features(self) -> List[str]:
        """所有特征"""
        return self.all_sparse_features + self.dense_features
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            'sparse_features': self.sparse_features,
            'dense_features': self.dense_features,
            'cross_features': self.cross_features,
            'exclude_features': self.exclude_features,
            'label_col': self.label_col,
            'high_cardinality_threshold': self.high_cardinality_threshold,
            'embed_dim': self.embed_dim,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'FeatureConfig':
        """从字典创建"""
        return cls(
            sparse_features=d.get('sparse_features', []),
            dense_features=d.get('dense_features', []),
            cross_features=[tuple(x) for x in d.get('cross_features', [])],
            exclude_features=d.get('exclude_features', []),
            label_col=d.get('label_col', 'label'),
            high_cardinality_threshold=d.get('high_cardinality_threshold', 50),
            embed_dim=d.get('embed_dim', 16),
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FeatureConfig':
        """从 YAML 文件加载配置
        
        Args:
            yaml_path: YAML 文件路径，支持相对路径（相对于 config 目录）
        
        Returns:
            FeatureConfig 实例
        """
        # 如果是相对路径，相对于 config 目录
        if not os.path.isabs(yaml_path):
            config_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(config_dir, yaml_path)
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        
        return cls.from_dict(d)
    
    def to_yaml(self, yaml_path: str) -> None:
        """保存配置到 YAML 文件
        
        Args:
            yaml_path: YAML 文件路径
        """
        d = self.to_dict()
        # 转换 tuple 为 list 以便 YAML 序列化
        d['cross_features'] = [list(x) for x in d['cross_features']]
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(d, f, allow_unicode=True, default_flow_style=False)


def load_default_config() -> FeatureConfig:
    """加载默认配置（从 features.yaml）"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(config_dir, 'features.yaml')
    
    if os.path.exists(yaml_path):
        return FeatureConfig.from_yaml(yaml_path)
    else:
        # 回退到旧的硬编码配置
        return FeatureConfig(
            sparse_features=[
                'business_type', 'offerid', 'country', 'bundle', 'adx',
                'make', 'model', 'demand_pkgname', 'campaignid',
            ],
            cross_features=[
                ('country', 'adx'),
                ('country', 'business_type'),
                ('make', 'model'),
                ('adx', 'business_type'),
            ],
        )


# 默认配置（从 YAML 加载）
DEFAULT_CONFIG = load_default_config()

# 精简配置（只用核心特征）
MINIMAL_CONFIG = FeatureConfig(
    sparse_features=['country', 'adx', 'business_type', 'campaignid'],
    cross_features=[('country', 'adx')],
)

# 基础特征配置（不含用户行为特征）
BASE_CONFIG = FeatureConfig(
    sparse_features=[
        'adid', 'adsize', 'adx', 'bundle', 'business_type',
        'campaignid', 'campaignsetid', 'carrier', 'city', 'connectiontype',
        'country', 'demand_pkgname', 'devicetype', 'image_id',
        'is_ifa_null', 'is_interstitial_ad', 'is_reward_ad',
        'language', 'make', 'model', 'offerid', 'os', 'osv',
        'publishier_id_new', 'rta_id', 'schain_new', 'screen_bucket',
        'source_adx_new', 'subcategory_id', 'tagid', 'tags', 'video_id',
    ],
    cross_features=[
        ('country', 'adx'),
        ('country', 'business_type'),
        ('make', 'model'),
        ('adx', 'business_type'),
    ],
)
