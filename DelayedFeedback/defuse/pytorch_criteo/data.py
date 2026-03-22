#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark - Data Processing

Implements data loading and preprocessing for all delayed feedback methods.
Aligned with TensorFlow implementation in src_tf/data.py.
"""
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass

from config import SECONDS_A_DAY, SECONDS_AN_HOUR


def load_criteo_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load Criteo conversion logs dataset.
    
    Returns:
        features: DataFrame with 17 feature columns (8 numeric + 9 categorical)
        click_ts: click timestamps
        pay_ts: conversion timestamps (-1 if no conversion)
    """
    df = pd.read_csv(data_path, sep="\t", header=None)
    
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()
    
    # Extract features
    features = df[df.columns[2:]]
    
    # Process categorical features (columns 8-16)
    for c in features.columns[8:]:
        features[c] = features[c].fillna("")
        features[c] = features[c].astype(str)
    
    # Process numeric features (columns 0-7): normalize to [0, 1]
    for c in features.columns[:8]:
        features[c] = features[c].fillna(-1)
        col_min, col_max = features[c].min(), features[c].max()
        if col_max > col_min:
            features[c] = (features[c] - col_min) / (col_max - col_min)
        else:
            features[c] = 0.0
    
    features.columns = [str(i) for i in range(17)]
    features.reset_index(drop=True, inplace=True)
    
    return features, click_ts, pay_ts


@dataclass
class DataDF:
    """
    Data container for delayed feedback experiments.
    Aligned with TF implementation.
    """
    x: pd.DataFrame           # Features
    click_ts: np.ndarray      # Click timestamps
    pay_ts: np.ndarray        # Conversion timestamps (-1 if no conversion)
    sample_ts: np.ndarray     # Sample timestamps (for streaming)
    labels: np.ndarray        # Observed labels
    delay_labels: Optional[np.ndarray] = None   # Delayed positive labels
    inw_labels: Optional[np.ndarray] = None     # In-window labels
    
    def __post_init__(self):
        if self.delay_labels is None:
            self.delay_labels = self.labels.copy()
        if self.inw_labels is None:
            self.inw_labels = self.labels.copy()
    
    def __len__(self):
        return len(self.labels)
    
    @classmethod
    def from_raw(cls, features: pd.DataFrame, click_ts: np.ndarray, 
                 pay_ts: np.ndarray, attr_win: Optional[int] = None) -> 'DataDF':
        """Create DataDF from raw data."""
        sample_ts = click_ts.copy()
        if attr_win is not None:
            labels = (np.logical_and(pay_ts > 0, pay_ts - click_ts < attr_win)).astype(np.int32)
        else:
            labels = (pay_ts > 0).astype(np.int32)
        return cls(features, click_ts, pay_ts, sample_ts, labels)
    
    def sub_days(self, start_day: int, end_day: int) -> 'DataDF':
        """Filter samples by day range based on sample_ts."""
        start_ts = start_day * SECONDS_A_DAY
        end_ts = end_day * SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts, self.sample_ts < end_ts)
        return DataDF(
            self.x.iloc[mask].reset_index(drop=True),
            self.click_ts[mask],
            self.pay_ts[mask],
            self.sample_ts[mask],
            self.labels[mask],
            self.delay_labels[mask] if self.delay_labels is not None else None,
            self.inw_labels[mask] if self.inw_labels is not None else None
        )
    
    def sub_hours(self, start_hour: int, end_hour: int) -> 'DataDF':
        """Filter samples by hour range based on sample_ts."""
        start_ts = start_hour * SECONDS_AN_HOUR
        end_ts = end_hour * SECONDS_AN_HOUR
        mask = np.logical_and(self.sample_ts >= start_ts, self.sample_ts < end_ts)
        return DataDF(
            self.x.iloc[mask].reset_index(drop=True),
            self.click_ts[mask],
            self.pay_ts[mask],
            self.sample_ts[mask],
            self.labels[mask],
            self.delay_labels[mask] if self.delay_labels is not None else None,
            self.inw_labels[mask] if self.inw_labels is not None else None
        )
    
    def form_vanilla(self, cut_size: int) -> 'DataDF':
        """
        Form vanilla dataset: delayed conversions become fake negatives.
        
        Args:
            cut_size: observation window in seconds
        """
        # Delayed positives become fake negatives
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels > 0)
        sample_ts = self.click_ts.copy()
        labels = self.labels.copy()
        labels[mask] = 0  # fake negatives
        
        # Sort by sample time
        idx = np.argsort(sample_ts)
        return DataDF(
            self.x.iloc[idx].reset_index(drop=True),
            self.click_ts[idx],
            self.pay_ts[idx],
            sample_ts[idx],
            labels[idx]
        )
    
    def form_oracle(self) -> 'DataDF':
        """
        Form oracle dataset: use true labels (no delayed feedback issue).
        This is the upper bound baseline.
        """
        idx = np.argsort(self.click_ts)
        return DataDF(
            self.x.iloc[idx].reset_index(drop=True),
            self.click_ts[idx],
            self.pay_ts[idx],
            self.click_ts[idx],
            self.labels[idx],
            self.labels[idx],
            self.labels[idx]
        )
    
    def add_fake_neg(self, attr_win: Optional[int] = None) -> 'DataDF':
        """
        Add fake negative samples for DFM.
        Positive samples are duplicated at pay_ts.
        """
        pos_mask = np.logical_and(self.pay_ts > 0, self.labels > 0)
        
        # Concatenate original + duplicated positives
        x = pd.concat([self.x, self.x.iloc[pos_mask]], ignore_index=True)
        sample_ts = np.concatenate([self.click_ts, self.pay_ts[pos_mask]])
        click_ts = np.concatenate([self.click_ts, self.click_ts[pos_mask]])
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask]])
        
        # Original positives become fake negatives
        labels = self.labels.copy()
        labels[pos_mask] = 0
        labels = np.concatenate([labels, np.ones(np.sum(pos_mask), dtype=np.int32)])
        
        # Sort by sample time
        idx = np.argsort(sample_ts)
        return DataDF(
            x.iloc[idx].reset_index(drop=True),
            click_ts[idx],
            pay_ts[idx],
            sample_ts[idx],
            labels[idx]
        )
    
    def add_esdfm_cut_fake_neg(self, cut_size: int) -> 'DataDF':
        """
        ES-DFM data processing: delayed positives are duplicated at pay_ts.
        
        Args:
            cut_size: observation window in seconds
        """
        # Delayed positives: conversion after cut_size
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels > 0)
        
        # Concatenate original + delayed positives
        x = pd.concat([self.x, self.x.iloc[mask]], ignore_index=True)
        sample_ts = np.concatenate([self.click_ts, self.pay_ts[mask]])
        click_ts = np.concatenate([self.click_ts, self.click_ts[mask]])
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]])
        
        # Original delayed positives become fake negatives
        labels = self.labels.copy()
        labels[mask] = 0
        labels = np.concatenate([labels, np.ones(np.sum(mask), dtype=np.int32)])
        
        # Sort by sample time
        idx = np.argsort(sample_ts)
        return DataDF(
            x.iloc[idx].reset_index(drop=True),
            click_ts[idx],
            pay_ts[idx],
            sample_ts[idx],
            labels[idx]
        )
    
    def add_inw_outw_delay_positive(self, cut_size: int) -> 'DataDF':
        """
        DEFUSE data processing: track in-window and delayed labels.
        
        Aligned with TF implementation:
        - Original samples: inw_label=1, delay_label=0
        - Duplicated DP samples: inw_label=0, delay_label=1
        
        Args:
            cut_size: observation window in seconds
        """
        # Delayed positives: conversion after cut_size
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels > 0)
        
        # Concatenate original + delayed positives (duplicated)
        x = pd.concat([self.x, self.x.iloc[mask]], ignore_index=True)
        sample_ts = np.concatenate([self.click_ts, self.pay_ts[mask]])
        click_ts = np.concatenate([self.click_ts, self.click_ts[mask]])
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]])
        
        # Labels: original delayed positives become fake negatives
        labels = self.labels.copy()
        labels[mask] = 0  # fake negatives
        labels = np.concatenate([labels, np.ones(np.sum(mask), dtype=np.int32)])
        
        # Delay labels: 1 for duplicated samples, 0 for original
        delay_labels = np.concatenate([
            np.zeros(len(self.labels), dtype=np.int32),
            np.ones(np.sum(mask), dtype=np.int32)
        ])
        
        # In-window labels: 1 - delay_labels (TF original)
        # Original samples: inw=1, Duplicated DP: inw=0
        inw_labels = 1 - delay_labels
        
        # Sort by sample time
        idx = np.argsort(sample_ts)
        return DataDF(
            x.iloc[idx].reset_index(drop=True),
            click_ts[idx],
            pay_ts[idx],
            sample_ts[idx],
            labels[idx],
            delay_labels[idx],
            inw_labels[idx]
        )
    
    def add_defer_duplicate_samples(self, cut_size: int, attr_win: int) -> 'DataDF':
        """
        DEFER data processing: duplicate samples strategy.
        
        Args:
            cut_size: observation window in seconds
            attr_win: attribution window in seconds
        """
        # In-window positives
        inw_mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= cut_size)
        # All positives (for duplication)
        label_mask = np.logical_and(self.pay_ts > 0, self.labels > 0)
        
        # Build 4 parts: inw_pos, others, dup_pos, dup_neg
        df1 = self.x.copy()  # observe data
        df2 = self.x.copy()  # duplicate data
        
        x = pd.concat([
            df1[inw_mask],
            df1[~inw_mask],
            df2[label_mask],
            df2[~label_mask]
        ], ignore_index=True)
        
        sample_ts = np.concatenate([
            self.click_ts[inw_mask],
            self.click_ts[~inw_mask],
            self.pay_ts[label_mask],
            self.click_ts[~label_mask] + attr_win
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_mask],
            self.click_ts[~inw_mask],
            self.click_ts[label_mask],
            self.click_ts[~label_mask]
        ])
        
        pay_ts = np.concatenate([
            self.pay_ts[inw_mask],
            self.pay_ts[~inw_mask],
            self.pay_ts[label_mask],
            self.pay_ts[~label_mask]
        ])
        
        labels = np.concatenate([
            np.ones(np.sum(inw_mask), dtype=np.int32),
            np.zeros(np.sum(~inw_mask), dtype=np.int32),
            self.labels[label_mask],
            self.labels[~label_mask]
        ])
        
        # Sort by sample time
        idx = np.argsort(sample_ts)
        return DataDF(
            x.iloc[idx].reset_index(drop=True),
            click_ts[idx],
            pay_ts[idx],
            sample_ts[idx],
            labels[idx]
        )
    
    def construct_tn_dp_data(self, cut_size: int, cut_day_sec: int) -> 'DataDF':
        """
        Construct data for tn/dp pretraining.
        
        Args:
            cut_size: observation window
            cut_day_sec: attribution window
        """
        # Delayed positives (beyond cut_day_sec)
        mask = self.pay_ts - self.click_ts > cut_day_sec
        
        # In-window positives
        inw_posmask = np.logical_and(
            self.pay_ts - self.click_ts <= cut_size,
            self.pay_ts > 0
        )
        
        # Delayed positives within attribution window
        delay_mask = np.logical_and(
            self.pay_ts - self.click_ts > cut_size,
            self.pay_ts - self.click_ts <= cut_day_sec
        )
        
        # Concatenate original + delayed positives
        x = pd.concat([self.x, self.x.iloc[mask]], ignore_index=True)
        sample_ts = np.concatenate([self.click_ts, self.pay_ts[mask]])
        click_ts = np.concatenate([self.click_ts, self.click_ts[mask]])
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]])
        
        # Labels
        labels = self.labels.copy()
        labels[mask] = 0
        
        # In-window labels
        inw_labels = self.labels.copy()
        inw_labels[~inw_posmask] = 0
        
        # Delay labels
        delay_labels = self.labels.copy()
        delay_labels[~delay_mask] = 0
        
        # Append for duplicated samples
        labels = np.concatenate([labels, np.ones(np.sum(mask), dtype=np.int32)])
        delay_labels = np.concatenate([delay_labels, np.ones(np.sum(mask), dtype=np.int32)])
        inw_labels = np.concatenate([inw_labels, np.zeros(np.sum(mask), dtype=np.int32)])
        
        # Sort by sample time
        idx = np.argsort(sample_ts)
        return DataDF(
            x.iloc[idx].reset_index(drop=True),
            click_ts[idx],
            pay_ts[idx],
            sample_ts[idx],
            labels[idx],
            delay_labels[idx],
            inw_labels[idx]
        )
    
    def to_tn_labels(self) -> np.ndarray:
        """Generate true negative labels for pretraining."""
        # True negative: no conversion OR conversion after 1 hour
        mask = np.logical_or(
            self.pay_ts < 0,
            self.pay_ts - self.click_ts > SECONDS_AN_HOUR
        )
        return (self.pay_ts < 0).astype(np.int32)
    
    def to_dp_labels(self) -> np.ndarray:
        """Generate delayed positive labels for pretraining."""
        # Delayed positive: conversion after 1 hour
        return (self.pay_ts - self.click_ts > SECONDS_AN_HOUR).astype(np.int32)


def get_streaming_data(
    data: DataDF,
    method: str,
    ob_win: int,
    attr_win: int,
    start_hour: int,
    end_hour: int
) -> Generator[Tuple[DataDF, DataDF], None, None]:
    """
    Generate streaming train/test data by hour.
    
    Args:
        data: Full dataset
        method: Method name (Vanilla, ES-DFM, DEFUSE, etc.)
        ob_win: Observation window in seconds
        attr_win: Attribution window in seconds
        start_hour: Start hour for streaming
        end_hour: End hour for streaming
    
    Yields:
        (train_data, test_data) for each hour
    """
    for hour in range(start_hour, end_hour):
        # Get data for this hour
        hour_data = data.sub_hours(hour, hour + 1)
        
        if len(hour_data) == 0:
            continue
        
        # Process according to method
        if method == "Vanilla":
            train_data = hour_data.form_vanilla(ob_win)
        elif method in ["FNW", "FNC"]:
            train_data = hour_data.form_vanilla(ob_win)
        elif method == "DFM":
            train_data = hour_data.add_fake_neg(attr_win)
        elif method == "DEFER":
            train_data = hour_data.add_defer_duplicate_samples(ob_win, attr_win)
        elif method == "ES-DFM":
            train_data = hour_data.add_esdfm_cut_fake_neg(ob_win)
        elif method in ["DEFUSE", "Bi-DEFUSE"]:
            train_data = hour_data.add_inw_outw_delay_positive(ob_win)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Test data uses oracle labels
        test_data = hour_data
        
        yield train_data, test_data

