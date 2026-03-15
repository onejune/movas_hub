#!/usr/bin/env python3
"""
将业务数据转换为 Defer 格式

输入格式 (CSV):
    req_hour,sample_bid_date,business_type,offerid,country,bundle,adx,make,model,demand_pkgname,campaignid,label,diff_hours

输出格式 (TSV):
    click_ts  pay_ts  num_0~num_7  cat_0~cat_8

用法:
    python convert_data.py --input_dir /path/to/data --output /path/to/output.txt
"""

import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import hashlib


def hash_feature(value, bins=512):
    """将类别特征哈希到固定范围"""
    if pd.isna(value) or value == '' or value == 'NULL':
        return 0
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % bins


def convert_data(input_paths, output_path, sample_rate=1.0):
    """
    转换数据格式
    
    Args:
        input_paths: 输入文件路径列表
        output_path: 输出文件路径
        sample_rate: 采样率 (0-1)
    """
    all_dfs = []
    
    print(f"共 {len(input_paths)} 个文件待处理")
    for i, input_path in enumerate(input_paths):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"读取文件 [{i+1}/{len(input_paths)}]: {input_path}")
        df = pd.read_csv(input_path)
        all_dfs.append(df)
    
    print("合并数据...")
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"总样本数: {len(df)}")
    
    # 采样
    if sample_rate < 1.0:
        df = df.sample(frac=sample_rate, random_state=42)
        print(f"采样后样本数: {len(df)}")
    
    # 1. 解析点击时间为时间戳
    df['click_ts'] = pd.to_datetime(df['req_hour'])
    base_time = df['click_ts'].min()
    print(f"时间范围: {df['click_ts'].min()} ~ {df['click_ts'].max()}")
    
    # 转换为相对秒数
    df['click_ts'] = (df['click_ts'] - base_time).dt.total_seconds().astype(int)
    
    # 2. 计算转化时间戳
    # diff_hours 为空、NULL、>=2400 或 label=0 表示未转化
    df['diff_hours'] = pd.to_numeric(df['diff_hours'], errors='coerce')
    
    is_converted = (
        (df['label'] == 1) & 
        (df['diff_hours'].notna()) & 
        (df['diff_hours'] < 2400)
    )
    
    df['pay_ts'] = np.where(
        is_converted,
        df['click_ts'] + df['diff_hours'] * 3600,
        -1
    )
    
    # 统计
    n_converted = is_converted.sum()
    n_total = len(df)
    print(f"转化样本: {n_converted} ({n_converted/n_total*100:.2f}%)")
    print(f"未转化样本: {n_total - n_converted} ({(n_total-n_converted)/n_total*100:.2f}%)")
    
    if n_converted > 0:
        delay_hours = df.loc[is_converted, 'diff_hours']
        print(f"转化延迟分布 (小时):")
        print(f"  均值: {delay_hours.mean():.2f}")
        print(f"  中位数: {delay_hours.median():.2f}")
        print(f"  15分钟内: {(delay_hours <= 0.25).mean()*100:.2f}%")
        print(f"  30分钟内: {(delay_hours <= 0.5).mean()*100:.2f}%")
        print(f"  1小时内: {(delay_hours <= 1).mean()*100:.2f}%")
        print(f"  24小时内: {(delay_hours <= 24).mean()*100:.2f}%")
    
    # 3. 构造数值特征 (8个)
    # 从时间中提取数值特征
    req_dt = pd.to_datetime(df['req_hour'])
    df['num_0'] = req_dt.dt.hour / 24.0  # 小时 (归一化)
    df['num_1'] = req_dt.dt.dayofweek / 7.0  # 星期几 (归一化)
    df['num_2'] = req_dt.dt.day / 31.0  # 日期 (归一化)
    df['num_3'] = req_dt.dt.month / 12.0  # 月份 (归一化)
    
    # 用 diff_hours 构造特征 (对于未转化样本用 0)
    df['num_4'] = np.where(is_converted, np.clip(df['diff_hours'] / 100, 0, 1), 0)
    
    # 填充剩余数值特征为随机值或 0
    df['num_5'] = np.random.random(len(df))
    df['num_6'] = np.random.random(len(df))
    df['num_7'] = np.random.random(len(df))
    
    # 4. 构造类别特征 (9个)
    cat_features = ['business_type', 'country', 'adx', 'make', 'bundle', 
                    'demand_pkgname', 'offerid', 'campaignid', 'model']
    
    cat_bins = [64, 256, 64, 256, 512, 256, 512, 512, 256]  # 每个特征的哈希桶数
    
    for i, (feat, bins) in enumerate(zip(cat_features, cat_bins)):
        if feat in df.columns:
            df[f'cat_{i}'] = df[feat].apply(lambda x: hash_feature(x, bins))
        else:
            df[f'cat_{i}'] = 0
    
    # 5. 按时间排序
    df = df.sort_values('click_ts')
    
    # 6. 输出
    output_cols = ['click_ts', 'pay_ts'] + \
                  [f'num_{i}' for i in range(8)] + \
                  [f'cat_{i}' for i in range(9)]
    
    output_df = df[output_cols]
    
    # 保存为 TSV 格式 (与 Criteo 格式一致)
    output_df.to_csv(output_path, sep='\t', header=False, index=False)
    
    print(f"\n输出文件: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # 计算天数
    total_seconds = df['click_ts'].max() - df['click_ts'].min()
    total_days = total_seconds / 86400
    print(f"时间跨度: {total_days:.1f} 天")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='转换业务数据为 Defer 格式')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入数据目录 (包含 sample_date=YYYY-MM-DD 子目录)')
    parser.add_argument('--output', type=str, required=True,
                        help='输出文件路径')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='采样率 (默认 1.0)')
    parser.add_argument('--dates', type=str, default=None,
                        help='指定日期范围, 如 "2026-03-04,2026-03-05" 或 "2026-03-04:2026-03-10"')
    
    args = parser.parse_args()
    
    # 查找所有数据文件
    input_dir = Path(args.input_dir)
    input_files = []
    
    # 遍历 sample_date=YYYY-MM-DD 目录
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith('sample_date='):
            date_str = subdir.name.split('=')[1]
            
            # 检查日期过滤
            if args.dates:
                if ':' in args.dates:
                    start_date, end_date = args.dates.split(':')
                    if not (start_date <= date_str <= end_date):
                        continue
                else:
                    allowed_dates = args.dates.split(',')
                    if date_str not in allowed_dates:
                        continue
            
            # 查找 CSV 文件
            for csv_file in subdir.glob('*.csv'):
                input_files.append(str(csv_file))
    
    if not input_files:
        print(f"错误: 在 {input_dir} 中未找到数据文件")
        return
    
    print(f"找到 {len(input_files)} 个数据文件")
    for f in input_files[:5]:
        print(f"  - {f}")
    if len(input_files) > 5:
        print(f"  ... 共 {len(input_files)} 个文件")
    
    convert_data(input_files, args.output, args.sample_rate)


if __name__ == '__main__':
    main()
