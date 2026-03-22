"""
DEFUSE 数据加载模块

支持:
- 天级增量加载 (避免 OOM)
- 特征编码
- 标签计算 (tn_label, dp_label, pos_label)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


class FeatureEncoder:
    """类别特征编码器"""
    
    def __init__(self):
        self.encoders: Dict[str, Dict] = {}
        self.vocab_sizes: Dict[str, int] = {}
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> 'FeatureEncoder':
        """拟合编码器 (增量更新)"""
        for col in tqdm(feature_cols, desc="Fitting encoder"):
            if col not in self.encoders:
                self.encoders[col] = {}
            
            unique_vals = df[col].unique()
            current_max = len(self.encoders[col])
            
            for val in unique_vals:
                if val not in self.encoders[col]:
                    self.encoders[col][val] = current_max + 1  # 0 留给 OOV
                    current_max += 1
            
            self.vocab_sizes[col] = current_max + 1
        
        return self
    
    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """转换特征为编码数组"""
        result = np.zeros((len(df), len(feature_cols)), dtype=np.int64)
        
        for i, col in enumerate(feature_cols):
            encoder = self.encoders.get(col, {})
            result[:, i] = df[col].map(lambda x: encoder.get(x, 0)).values
        
        return result
    
    def save(self, path: str):
        """保存编码器"""
        # 转换所有 key 为 str (JSON 要求)
        encoders_str = {
            col: {str(k): v for k, v in enc.items()}
            for col, enc in self.encoders.items()
        }
        with open(path, 'w') as f:
            json.dump({
                'encoders': encoders_str,
                'vocab_sizes': self.vocab_sizes
            }, f)
    
    def load(self, path: str) -> 'FeatureEncoder':
        """加载编码器"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.encoders = {col: {k: v for k, v in enc.items()} 
                           for col, enc in data['encoders'].items()}
            self.vocab_sizes = data['vocab_sizes']
        return self
    
    def get_vocab_sizes_list(self, feature_cols: List[str]) -> List[int]:
        """返回 vocab_sizes 列表"""
        return [self.vocab_sizes[col] for col in feature_cols]


def get_date_range(start_date: str, end_date: str) -> List[str]:
    """生成日期范围"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates


def list_available_dates(data_dir: str) -> List[str]:
    """列出可用的日期分区"""
    dates = []
    for d in Path(data_dir).iterdir():
        if d.is_dir() and d.name.startswith("sample_date="):
            date_str = d.name.split("=")[1]
            dates.append(date_str)
    return sorted(dates)


def load_day_data(data_dir: str, date: str) -> pd.DataFrame:
    """加载单天数据"""
    path = Path(data_dir) / f"sample_date={date}"
    if not path.exists():
        return pd.DataFrame()
    
    df = pq.read_table(path).to_pandas()
    df['sample_date'] = date
    return df


def compute_labels(df: pd.DataFrame, 
                   observation_window: int = 168,
                   attribution_window: int = 24) -> pd.DataFrame:
    """
    计算 DEFUSE 所需的标签
    
    Args:
        df: 原始数据
        observation_window: 观察窗口 (小时), 默认 168h (7天)
        attribution_window: 归因窗口 (小时), 默认 24h
    
    Returns:
        添加了标签列的 DataFrame:
        - label_oracle: 观察窗口内是否转化 (用于评估)
        - pos_label: 归因窗口内是否转化 (用于训练)
        - tn_label: 是否为真负样本 (观察窗口内未转化)
        - dp_label: 是否为延迟正样本 (归因窗口外转化)
        - delay_time: 转化延迟时间 (小时)
        - elapsed_time: 观察时间 (小时)
    """
    df = df.copy()
    
    # diff_hours: 转化发生的时间差 (小时), 未转化则为很大的值
    diff_hours = df['diff_hours'].fillna(observation_window * 10).values
    
    # label_oracle: 观察窗口内转化
    df['label_oracle'] = (diff_hours <= observation_window).astype(np.float32)
    
    # pos_label: 归因窗口内转化 (用于训练)
    df['pos_label'] = (diff_hours <= attribution_window).astype(np.float32)
    
    # tn_label: 真负样本 (观察窗口内未转化)
    df['tn_label'] = (diff_hours > observation_window).astype(np.float32)
    
    # dp_label: 延迟正样本 (归因窗口外但观察窗口内转化)
    df['dp_label'] = ((diff_hours > attribution_window) & 
                      (diff_hours <= observation_window)).astype(np.float32)
    
    # delay_time: 转化延迟时间
    # 正样本: 实际转化时间
    # 负样本: 使用观察窗口 (表示至少等了这么久没转化)
    df['delay_time'] = np.where(
        df['label_oracle'] == 1,
        np.clip(diff_hours, 0, observation_window),
        observation_window
    ).astype(np.float32)
    
    # elapsed_time: 观察时间 (对于这个数据集, 都是完整观察窗口)
    df['elapsed_time'] = observation_window
    
    return df


def iterate_days(data_dir: str, 
                 dates: List[str],
                 days_per_batch: int = 7,
                 exclude_cols: List[str] = None) -> Iterator[Tuple[pd.DataFrame, List[str]]]:
    """
    天级增量迭代器
    
    Args:
        data_dir: 数据目录
        dates: 日期列表
        days_per_batch: 每批加载多少天
        exclude_cols: 排除的列
    
    Yields:
        (df, feature_cols): 数据和特征列
    """
    exclude_cols = exclude_cols or []
    feature_cols = None
    
    for i in range(0, len(dates), days_per_batch):
        batch_dates = dates[i:i + days_per_batch]
        
        dfs = []
        for date in tqdm(batch_dates, desc=f"Loading days {i+1}-{i+len(batch_dates)}"):
            df = load_day_data(data_dir, date)
            if len(df) > 0:
                dfs.append(df)
        
        if not dfs:
            continue
        
        batch_df = pd.concat(dfs, ignore_index=True)
        
        # 计算标签
        batch_df = compute_labels(batch_df)
        
        # 确定特征列 (第一批时确定)
        if feature_cols is None:
            all_cols = batch_df.columns.tolist()
            label_cols = ['label', 'label_oracle', 'pos_label', 'tn_label', 
                         'dp_label', 'delay_time', 'elapsed_time', 'diff_hours']
            feature_cols = [c for c in all_cols 
                          if c not in exclude_cols and c not in label_cols]
        
        yield batch_df, feature_cols


class IncrementalDataLoader:
    """
    天级增量数据加载器
    
    用于 full data 训练, 避免一次性加载所有数据导致 OOM
    """
    
    def __init__(self, 
                 data_dir: str,
                 train_dates: List[str],
                 test_dates: List[str],
                 encoder: Optional[FeatureEncoder] = None,
                 exclude_cols: List[str] = None,
                 days_per_batch: int = 7):
        self.data_dir = data_dir
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.encoder = encoder or FeatureEncoder()
        self.exclude_cols = exclude_cols or []
        self.days_per_batch = days_per_batch
        self.feature_cols = None
        self._fitted = False
    
    def fit_encoder(self) -> 'IncrementalDataLoader':
        """拟合编码器 (遍历所有训练数据)"""
        print("Fitting encoder on training data...")
        
        for batch_df, feature_cols in iterate_days(
            self.data_dir, 
            self.train_dates,
            self.days_per_batch,
            self.exclude_cols
        ):
            if self.feature_cols is None:
                self.feature_cols = feature_cols
            
            self.encoder.fit(batch_df, feature_cols)
        
        self._fitted = True
        print(f"Encoder fitted: {len(self.feature_cols)} features, "
              f"total vocab size: {sum(self.encoder.vocab_sizes.values())}")
        
        return self
    
    def iterate_train(self) -> Iterator[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        迭代训练数据
        
        Yields:
            (X, y_dict): 编码后的特征和标签字典
        """
        if not self._fitted:
            self.fit_encoder()
        
        for batch_df, _ in iterate_days(
            self.data_dir,
            self.train_dates,
            self.days_per_batch,
            self.exclude_cols
        ):
            X = self.encoder.transform(batch_df, self.feature_cols)
            y = {
                'label': batch_df['pos_label'].values.astype(np.float32),
                'tn_label': batch_df['tn_label'].values.astype(np.float32),
                'dp_label': batch_df['dp_label'].values.astype(np.float32),
                'delay_time': batch_df['delay_time'].values.astype(np.float32),
            }
            yield X, y
    
    def load_test(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """加载测试数据 (一次性)"""
        if not self._fitted:
            self.fit_encoder()
        
        dfs = []
        for date in tqdm(self.test_dates, desc="Loading test data"):
            df = load_day_data(self.data_dir, date)
            if len(df) > 0:
                dfs.append(df)
        
        test_df = pd.concat(dfs, ignore_index=True)
        test_df = compute_labels(test_df)
        
        X = self.encoder.transform(test_df, self.feature_cols)
        y = {
            'label': test_df['label_oracle'].values.astype(np.float32),  # 用 oracle 评估
        }
        
        return X, y
    
    def get_vocab_sizes(self) -> List[int]:
        """获取 vocab sizes"""
        return self.encoder.get_vocab_sizes_list(self.feature_cols)
    
    def save_encoder(self, path: str):
        """保存编码器"""
        self.encoder.save(path)
        
        # 同时保存 feature_cols
        meta_path = Path(path).parent / "feature_cols.json"
        with open(meta_path, 'w') as f:
            json.dump(self.feature_cols, f)
    
    def load_encoder(self, path: str) -> 'IncrementalDataLoader':
        """加载编码器"""
        self.encoder.load(path)
        
        meta_path = Path(path).parent / "feature_cols.json"
        with open(meta_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        self._fitted = True
        return self
