"""
DEFUSE 配置文件
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    raw_data_dir: str = "/mnt/data/oss_wanjun/pai_work/defer_sample_parquet"
    processed_data_dir: str = "/mnt/workspace/walter.wan/open_research/DEFUSE/data"
    
    # 时间窗口 (小时)
    observation_window: int = 168  # 7 天观察窗口
    attribution_window: int = 24   # 24 小时归因窗口 (DEFUSE 单窗口)
    
    # 训练/测试划分
    train_end_date: str = "2026-02-28"  # 训练集截止日期
    test_start_date: str = "2026-03-01"  # 测试集开始日期
    
    # 特征列 (排除 label 相关列)
    exclude_cols: List[str] = field(default_factory=lambda: [
        'req_hour', 'sample_bid_date', 'label', 'diff_hours',
        'sample_date',  # partition column
    ])


@dataclass
class ModelConfig:
    """模型配置"""
    embed_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 4096
    learning_rate: float = 0.001
    pretrain_epochs: int = 1
    finetune_epochs: int = 1
    num_workers: int = 4
    
    # 天级增量训练
    incremental: bool = True
    days_per_batch: int = 7  # 每次加载多少天数据
    
    # 输出目录
    output_dir: str = "/mnt/workspace/walter.wan/open_research/DEFUSE/outputs"
    
    # 方法: defuse, esdfm
    methods: List[str] = field(default_factory=lambda: ['defuse', 'esdfm'])


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def __post_init__(self):
        # 确保目录存在
        Path(self.data.processed_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.train.output_dir).mkdir(parents=True, exist_ok=True)
