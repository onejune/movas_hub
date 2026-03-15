#!/usr/bin/env python3
"""
快速数据转换 - 多进程并行版本

只使用原始特征，不添加任何衍生特征
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')


def hash_feature(value, bins=512):
    """将类别特征哈希到固定范围"""
    if pd.isna(value) or value == '' or value == 'NULL':
        return 0
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % bins


def process_single_file(file_path, cat_bins, cat_features):
    """处理单个文件，返回转换后的数据"""
    try:
        df = pd.read_csv(file_path)
        if len(df) == 0:
            return None
        
        # 解析点击时间
        df['click_ts'] = pd.to_datetime(df['req_hour'])
        
        # 计算转化时间戳
        df['diff_hours'] = pd.to_numeric(df['diff_hours'], errors='coerce')
        is_converted = (
            (df['label'] == 1) & 
            (df['diff_hours'].notna()) & 
            (df['diff_hours'] < 2400) &
            (df['diff_hours'] > -1000)
        )
        
        # 数值特征: 只用时间相关的 (请求时刻可知)
        req_dt = df['click_ts']
        num_feats = np.column_stack([
            req_dt.dt.hour.values / 24.0,        # 小时
            req_dt.dt.dayofweek.values / 7.0,    # 星期几
            req_dt.dt.day.values / 31.0,         # 日期
            req_dt.dt.month.values / 12.0,       # 月份
            req_dt.dt.minute.values / 60.0,      # 分钟
            req_dt.dt.second.values / 60.0,      # 秒
            (req_dt.dt.hour.values >= 9) & (req_dt.dt.hour.values <= 21),  # 是否工作时间
            req_dt.dt.dayofweek.values < 5,      # 是否工作日
        ]).astype(np.float32)
        
        # 类别特征: 只用原始字段
        cat_feats = np.zeros((len(df), len(cat_features)), dtype=np.int32)
        for i, (feat, bins) in enumerate(zip(cat_features, cat_bins)):
            if feat in df.columns:
                cat_feats[:, i] = df[feat].apply(lambda x: hash_feature(x, bins)).values
        
        # 时间戳 (相对于 2026-01-01)
        base_time = pd.Timestamp('2026-01-01')
        click_ts = (df['click_ts'] - base_time).dt.total_seconds().values.astype(np.int64)
        pay_ts = np.where(
            is_converted,
            click_ts + df['diff_hours'].values * 3600,
            -1
        )
        
        return {
            'click_ts': click_ts,
            'pay_ts': pay_ts,
            'num_feats': num_feats,
            'cat_feats': cat_feats,
            'n_converted': is_converted.sum(),
            'n_total': len(df)
        }
    except Exception as e:
        print(f"处理文件失败 {file_path}: {e}")
        return None


def convert_data_parallel(input_paths, output_path, n_workers=None):
    """并行转换数据"""
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    
    # 类别特征配置 (只用原始字段)
    cat_features = ['business_type', 'country', 'adx', 'make', 'bundle', 
                    'demand_pkgname', 'offerid', 'campaignid', 'model']
    cat_bins = [64, 256, 64, 256, 512, 256, 512, 512, 256]
    
    print(f"使用 {n_workers} 个进程并行处理 {len(input_paths)} 个文件...")
    print(f"类别特征: {cat_features}")
    print(f"数值特征: hour, dayofweek, day, month, minute, second, is_work_hour, is_weekday")
    
    # 并行处理
    process_fn = partial(process_single_file, cat_bins=cat_bins, cat_features=cat_features)
    
    all_data = []
    total_converted = 0
    total_samples = 0
    
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_fn, input_paths, chunksize=50)):
            if result is not None:
                all_data.append(result)
                total_converted += result['n_converted']
                total_samples += result['n_total']
            
            if (i + 1) % 500 == 0:
                print(f"已处理 {i+1}/{len(input_paths)} 个文件, 累计 {total_samples:,} 条样本")
    
    print(f"\n合并数据...")
    
    # 合并所有数据
    click_ts = np.concatenate([d['click_ts'] for d in all_data])
    pay_ts = np.concatenate([d['pay_ts'] for d in all_data])
    num_feats = np.vstack([d['num_feats'] for d in all_data])
    cat_feats = np.vstack([d['cat_feats'] for d in all_data])
    
    print(f"总样本数: {len(click_ts):,}")
    print(f"转化样本: {total_converted:,} ({total_converted/len(click_ts)*100:.2f}%)")
    
    # 按时间排序
    print("按时间排序...")
    sort_idx = np.argsort(click_ts)
    click_ts = click_ts[sort_idx]
    pay_ts = pay_ts[sort_idx]
    num_feats = num_feats[sort_idx]
    cat_feats = cat_feats[sort_idx]
    
    # 时间范围
    total_days = (click_ts.max() - click_ts.min()) / 86400
    print(f"时间跨度: {total_days:.1f} 天")
    
    # 写入文件
    print(f"写入文件: {output_path}")
    with open(output_path, 'w') as f:
        for i in range(len(click_ts)):
            row = [str(click_ts[i]), str(pay_ts[i])]
            row.extend([f"{x:.6f}" for x in num_feats[i]])
            row.extend([str(x) for x in cat_feats[i]])
            f.write('\t'.join(row) + '\n')
            
            if (i + 1) % 1000000 == 0:
                print(f"已写入 {i+1:,} 条")
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n完成! 文件大小: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='快速转换业务数据为 Defer 格式')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--workers', type=int, default=None)
    
    args = parser.parse_args()
    
    # 收集所有文件
    input_dir = Path(args.input_dir)
    input_files = []
    
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith('sample_date='):
            for csv_file in subdir.glob('*.csv'):
                input_files.append(str(csv_file))
    
    print(f"找到 {len(input_files)} 个数据文件")
    
    convert_data_parallel(input_files, args.output, args.workers)


if __name__ == '__main__':
    main()
