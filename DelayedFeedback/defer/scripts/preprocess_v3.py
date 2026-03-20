#!/usr/bin/env python3
"""
Defer 数据预处理脚本 v3
- 新增 DFM/ES-DFM 所需的 delay_time, elapsed_time, dp_label 等列
- 输入: /mnt/data/oss_wanjun/pai_work/defer_sample_parquet/
- 输出: /mnt/workspace/walter.wan/open_research/defer/data_v3/
"""

import os
import json
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 配置
INPUT_DIR = "/mnt/data/oss_wanjun/pai_work/defer_sample_parquet"
OUTPUT_DIR = "/mnt/workspace/walter.wan/open_research/defer/data_v3"

# 排除列（非特征）
EXCLUDE_COLS = ['req_hour', 'sample_bid_date', 'sample_date', 'label', 'diff_hours']

# 时间窗口（小时）
TIME_WINDOWS = [24, 48, 72, 168]

# 数据划分
TRAIN_END_DATE = "2026-03-01"  # 训练集截止日期（含）

# DFM 参数
MAX_OBSERVATION_WINDOW = 168  # 最大观察窗口（小时）= 7 天
DEFAULT_ELAPSED_TIME = 168  # 负样本的默认 elapsed_time


def get_feature_cols(sample_df):
    """获取特征列"""
    all_cols = sample_df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in EXCLUDE_COLS]
    return feature_cols


def process_partition(partition_path, feature_cols):
    """处理单个分区"""
    df = pq.read_table(partition_path).to_pandas()
    
    # 提取日期
    date_str = os.path.basename(partition_path).split('=')[1]
    
    # 特征预处理：空值填充
    for col in feature_cols:
        df[col] = df[col].fillna('none').replace('', 'none').replace('None', 'none').astype(str)
    
    # ========== 基础标签 ==========
    df['label_oracle'] = df['label'].astype(int)
    
    # diff_hours: 点击到转化的时间（小时），NaN 表示未转化
    # 原始数据中 diff_hours=2400 表示未转化（超过观察窗口）
    diff_hours = df['diff_hours'].fillna(2400.0)
    
    # ========== 时间窗口标签 ==========
    for w in TIME_WINDOWS:
        # 窗口内是否转化
        df[f'label_{w}h'] = ((df['label'] == 1) & (diff_hours <= w)).astype(int)
        # 窗口是否可观察（负样本或窗口内转化）
        df[f'observable_{w}h'] = (
            (df['label'] == 0) | 
            ((df['label'] == 1) & (diff_hours <= w))
        ).astype(int)
    
    # ========== DFM 所需标签 ==========
    # delay_time: 从点击到转化的时间（小时）
    # - 正样本: 实际转化时间 (diff_hours)
    # - 负样本: 0 (未转化)
    df['delay_time'] = np.where(
        df['label'] == 1,
        np.clip(diff_hours, 0, MAX_OBSERVATION_WINDOW),  # 正样本: 实际转化时间，截断到最大窗口
        0.0  # 负样本: 0
    ).astype(np.float32)
    
    # elapsed_time: 从点击到数据截止的时间（小时）
    # - 正样本: 转化时间 (delay_time)
    # - 负样本: 观察窗口时间 (假设所有负样本都观察了完整窗口)
    df['elapsed_time'] = np.where(
        df['label'] == 1,
        df['delay_time'],  # 正样本: 转化时间
        DEFAULT_ELAPSED_TIME  # 负样本: 默认观察窗口
    ).astype(np.float32)
    
    # ========== ES-DFM 所需标签 ==========
    # tn_label: 真负样本 (true negative) - 观察窗口结束后仍未转化
    df['tn_label'] = (df['label'] == 0).astype(int)
    
    # dp_label: 延迟正样本 (delayed positive) - 在某个窗口外转化
    # 使用 24h 窗口作为基准：24h 内未转化但最终转化的样本
    df['dp_label'] = (
        (df['label'] == 1) & (diff_hours > 24)
    ).astype(int)
    
    # pos_label: 窗口内正样本 - 24h 内转化
    df['pos_label'] = (
        (df['label'] == 1) & (diff_hours <= 24)
    ).astype(int)
    
    # ========== 输出列 ==========
    label_cols = (
        ['label_oracle'] + 
        [f'label_{w}h' for w in TIME_WINDOWS] + 
        [f'observable_{w}h' for w in TIME_WINDOWS] +
        ['delay_time', 'elapsed_time', 'tn_label', 'dp_label', 'pos_label']
    )
    
    output_cols = feature_cols + label_cols
    
    return df[output_cols], date_str


def main():
    print("=" * 60)
    print("Defer 数据预处理 v3 (含 DFM/ES-DFM 标签)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有分区
    partitions = sorted([d for d in os.listdir(INPUT_DIR) if d.startswith('sample_date=')])
    print(f"发现 {len(partitions)} 个分区")
    
    # 获取特征列（从第一个分区采样）
    sample_df = pq.read_table(os.path.join(INPUT_DIR, partitions[0])).to_pandas()
    feature_cols = get_feature_cols(sample_df)
    print(f"特征列数: {len(feature_cols)}")
    del sample_df
    
    # 分批处理，按日期划分
    train_dfs = []
    test_dfs = []
    
    for p in tqdm(partitions, desc="处理分区"):
        partition_path = os.path.join(INPUT_DIR, p)
        df, date_str = process_partition(partition_path, feature_cols)
        
        if date_str <= TRAIN_END_DATE:
            train_dfs.append(df)
        else:
            test_dfs.append(df)
    
    # 合并并保存训练集
    print("合并训练集...")
    train_df = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    print(f"训练集: {len(train_df):,} 行")
    
    # 统计信息
    train_rows = len(train_df)
    train_cvr = float(train_df['label_oracle'].mean())
    train_cvr_by_window = {f'{w}h': float(train_df[f'label_{w}h'].mean()) for w in TIME_WINDOWS}
    train_dp_rate = float(train_df['dp_label'].mean())
    train_pos_rate = float(train_df['pos_label'].mean())
    train_delay_time_mean = float(train_df[train_df['label_oracle'] == 1]['delay_time'].mean())
    
    print("保存训练集...")
    train_df.to_parquet(
        os.path.join(OUTPUT_DIR, 'train.parquet'),
        index=False,
        compression='snappy'
    )
    del train_df
    
    # 合并并保存测试集
    print("合并测试集...")
    test_df = pd.concat(test_dfs, ignore_index=True)
    del test_dfs
    print(f"测试集: {len(test_df):,} 行")
    
    test_rows = len(test_df)
    test_cvr = float(test_df['label_oracle'].mean())
    test_cvr_by_window = {f'{w}h': float(test_df[f'label_{w}h'].mean()) for w in TIME_WINDOWS}
    
    print("保存测试集...")
    test_df.to_parquet(
        os.path.join(OUTPUT_DIR, 'test.parquet'),
        index=False,
        compression='snappy'
    )
    del test_df
    
    # 标签列
    label_cols = (
        ['label_oracle'] + 
        [f'label_{w}h' for w in TIME_WINDOWS] + 
        [f'observable_{w}h' for w in TIME_WINDOWS] +
        ['delay_time', 'elapsed_time', 'tn_label', 'dp_label', 'pos_label']
    )
    
    # 保存特征列名
    with open(os.path.join(OUTPUT_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    # 保存标签列名
    with open(os.path.join(OUTPUT_DIR, 'label_cols.json'), 'w') as f:
        json.dump(label_cols, f, indent=2)
    
    # 保存元信息
    meta = {
        'train_rows': train_rows,
        'test_rows': test_rows,
        'num_features': len(feature_cols),
        'time_windows': TIME_WINDOWS,
        'train_end_date': TRAIN_END_DATE,
        'train_cvr': train_cvr,
        'test_cvr': test_cvr,
        'train_cvr_by_window': train_cvr_by_window,
        'test_cvr_by_window': test_cvr_by_window,
        'train_dp_rate': train_dp_rate,
        'train_pos_rate': train_pos_rate,
        'train_delay_time_mean': train_delay_time_mean,
        'max_observation_window': MAX_OBSERVATION_WINDOW,
    }
    with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # 打印汇总
    print()
    print("=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"训练集: {train_rows:,} 行, CVR={train_cvr*100:.2f}%")
    print(f"测试集: {test_rows:,} 行, CVR={test_cvr*100:.2f}%")
    print(f"特征数: {len(feature_cols)}")
    print()
    print("各窗口转化率 (训练集):")
    for w, cvr in train_cvr_by_window.items():
        print(f"  {w}: {cvr*100:.2f}%")
    print()
    print("DFM/ES-DFM 统计 (训练集):")
    print(f"  延迟正样本率 (dp_label): {train_dp_rate*100:.2f}%")
    print(f"  窗口内正样本率 (pos_label): {train_pos_rate*100:.2f}%")
    print(f"  正样本平均转化时间: {train_delay_time_mean:.1f}h")
    print()
    print(f"数据已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
