#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark - Configuration
"""
from dataclasses import dataclass, field
from typing import List


# Time constants
SECONDS_A_DAY = 60 * 60 * 24
SECONDS_AN_HOUR = 60 * 60


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = "/mnt/workspace/walter.wan/open_research/criteo_dataset/data.txt"
    output_dir: str = "/mnt/workspace/walter.wan/open_research/DEFUSE/outputs_criteo"
    
    # Time windows (seconds)
    observation_window: int = SECONDS_AN_HOUR  # 1 hour
    attribution_window: int = 7 * SECONDS_A_DAY  # 7 days
    
    # Train/test split (days)
    train_start_day: int = 0
    train_end_day: int = 60
    test_start_day: int = 30
    test_end_day: int = 60
    
    # Feature config
    num_features: int = 8
    cat_features: int = 9
    num_bin_sizes: tuple = (64, 16, 128, 64, 128, 64, 512, 512)
    cat_bin_sizes: tuple = (512, 128, 256, 256, 64, 256, 256, 16, 256)
    embed_dim: int = 8


@dataclass
class ModelConfig:
    """Model configuration"""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    dropout: float = 0.0
    l2_reg: float = 1e-5
    use_batch_norm: bool = True


@dataclass
class TrainConfig:
    """Training configuration"""
    batch_size: int = 1024
    learning_rate: float = 0.001
    optimizer: str = "adam"
    pretrain_epochs: int = 1
    finetune_epochs: int = 1
    num_workers: int = 4
    
    # Streaming training
    streaming: bool = True
    hours_per_batch: int = 1


@dataclass
class Config:
    """Combined configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # Available methods
    methods: List[str] = field(default_factory=lambda: [
        "Vanilla", "FNW", "FNC", "DFM", 
        "DEFER", "ES-DFM", "DEFUSE", "Bi-DEFUSE"
    ])
