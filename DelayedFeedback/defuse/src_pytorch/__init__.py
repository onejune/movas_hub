"""
DEFUSE PyTorch Implementation

模块结构:
- config.py: 配置
- data.py: 数据加载 (支持天级增量)
- models.py: 模型定义
- loss.py: 损失函数 (DEFUSE, ES-DFM)
- trainer.py: 训练器
"""

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .data import (
    FeatureEncoder, IncrementalDataLoader, compute_labels,
    list_available_dates, get_date_range
)
from .models import DEFUSEModel
from .loss import defuse_loss, esdfm_loss, pretrain_loss
from .trainer import Trainer

__all__ = [
    'Config', 'DataConfig', 'ModelConfig', 'TrainConfig',
    'FeatureEncoder', 'IncrementalDataLoader', 'compute_labels',
    'list_available_dates', 'get_date_range',
    'DEFUSEModel',
    'defuse_loss', 'esdfm_loss', 'pretrain_loss',
    'Trainer',
]
