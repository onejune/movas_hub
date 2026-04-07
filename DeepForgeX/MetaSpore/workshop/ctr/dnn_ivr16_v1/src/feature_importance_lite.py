#!/usr/bin/env python3
"""
特征重要性评估 - 轻量版 (纯 Pandas，无需 Spark)
只读需要的列，内存友好
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import glob
import json
from collections import defaultdict

# 配置
DATA_PATH = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet"
SCHEMA_PATH = "./conf/combine_schema"
OUTPUT_DIR = "./output/feature_importance"
SAMPLE_DATE = "2026-03-03"
NUM_FILES = 3  # 读取的 parquet 文件数


def load_features(schema_path):
    """加载特征列表"""
    features = []
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def load_data(data_path, date, features, num_files=3):
    """只读需要的列"""
    print(f"读取数据: {data_path}/{date}")
    
    # 提取基础特征名
    base_features = list(set([f.split('#')[0] for f in features]))
    base_features.append('label')
    
    files = sorted(glob.glob(f"{data_path}/{date}/part-*.parquet"))[:num_files]
    print(f"读取 {len(files)} 个文件")
    
    dfs = []
    for f in files:
        pf = pq.ParquetFile(f)
        available = set(pf.schema.names)
        cols = [c for c in base_features if c in available]
        df = pq.read_table(f, columns=cols).to_pandas()
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"总行数: {len(df):,}, 列数: {len(df.columns)}")
    print(f"内存: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df


def analyze_coverage(df, features):
    """分析覆盖率和基数"""
    print("\n=== 覆盖率分析 ===")
    results = []
    n = len(df)
    
    for i, feat in enumerate(features):
        base_feat = feat.split('#')[0]
        
        if base_feat not in df.columns:
            results.append({'feature': feat, 'coverage': 0, 'cardinality': 0, 'status': 'NOT_IN_DATA'})
            continue
        
        col = df[base_feat]
        
        # 覆盖率
        if col.dtype == 'object':
            non_null = ((col.notna()) & (col != '') & (col != 'null') & (col != 'None')).sum()
        else:
            non_null = col.notna().sum()
        
        coverage = non_null / n
        cardinality = col.nunique()
        
        # 状态
        if coverage < 0.01:
            status = 'SPARSE'
        elif coverage < 0.1:
            status = 'LOW_COV'
        elif cardinality > 500000:
            status = 'HIGH_CARD'
        elif coverage > 0.5:
            status = 'HIGH_VALUE'
        else:
            status = 'MEDIUM'
        
        results.append({
            'feature': feat,
            'coverage': round(coverage, 4),
            'cardinality': int(cardinality),
            'status': status
        })
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(features)}")
    
    return pd.DataFrame(results)


def analyze_lift(df, features):
    """分析 Lift"""
    print("\n=== Lift 分析 ===")
    
    pos_rate = df['label'].mean()
    print(f"整体正样本率: {pos_rate:.4f}")
    
    results = []
    
    for i, feat in enumerate(features):
        base_feat = feat.split('#')[0]
        
        if base_feat not in df.columns:
            results.append({'feature': feat, 'max_lift': 0, 'min_lift': 0, 'lift_range': 0})
            continue
        
        try:
            grp = df.groupby(base_feat)['label'].agg(['mean', 'count'])
            grp = grp[grp['count'] >= 50]
            
            if len(grp) < 2:
                results.append({'feature': feat, 'max_lift': 1, 'min_lift': 1, 'lift_range': 0})
                continue
            
            rates = grp['mean'].dropna()
            max_l = rates.max() / pos_rate if pos_rate > 0 else 1
            min_l = rates.min() / pos_rate if pos_rate > 0 else 1
            
            results.append({
                'feature': feat,
                'max_lift': round(max_l, 4),
                'min_lift': round(min_l, 4),
                'lift_range': round(max_l - min_l, 4)
            })
        except:
            results.append({'feature': feat, 'max_lift': 0, 'min_lift': 0, 'lift_range': 0})
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(features)}")
    
    return pd.DataFrame(results)


def analyze_redundancy(features):
    """时间窗口冗余分析"""
    print("\n=== 冗余分析 ===")
    
    time_windows = ['1d', '3d', '7d', '14d', '15d', '30d', '60d', '90d', '180d',
                    '1h', '3h', '12h', '24h', '48h', '3m', '10m', '30m',
                    '1_3d', '4_10d', '11_30d', '31_60d', '61_90d', '61_180d']
    
    # 推荐保留
    keep_windows = {'1d', '3d', '7d', '14d', '30d', '1h', '3h', '24h', '48h', '3m', '30m', '1_3d', '4_10d', '11_30d'}
    
    groups = defaultdict(list)
    
    for feat in features:
        base = feat.split('#')[0]
        for tw in time_windows:
            if f'_{tw}' in base:
                grp = base.replace(f'_{tw}', '_*')
                groups[grp].append((feat, tw))
                break
    
    redundant = []
    for grp, members in groups.items():
        if len(members) > 3:
            for feat, tw in members:
                if tw not in keep_windows:
                    redundant.append(feat)
    
    print(f"时间窗口组: {len(groups)}, 冗余特征: {len(redundant)}")
    return {'groups': dict(groups), 'redundant': redundant}


def final_score(coverage_df, lift_df, redundancy):
    """综合评分"""
    print("\n=== 综合评分 ===")
    
    df = coverage_df.merge(lift_df, on='feature', how='outer')
    redundant_set = set(redundancy.get('redundant', []))
    df['is_redundant'] = df['feature'].isin(redundant_set)
    
    def calc(row):
        cov = row.get('coverage', 0)
        cov_s = 100 if cov >= 0.5 else (70 if cov >= 0.1 else (40 if cov >= 0.01 else 10))
        
        lift = row.get('lift_range', 0)
        lift_s = 100 if lift >= 2 else (80 if lift >= 1 else (60 if lift >= 0.5 else (40 if lift >= 0.1 else 20)))
        
        penalty = -30 if row.get('is_redundant') else 0
        return max(0, min(100, cov_s * 0.3 + lift_s * 0.5 + penalty))
    
    df['score'] = df.apply(calc, axis=1)
    df['action'] = df['score'].apply(lambda s: 'KEEP' if s >= 55 else ('REVIEW' if s >= 35 else 'REMOVE'))
    
    return df.sort_values('score', ascending=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 50)
    print("特征重要性评估")
    print("=" * 50)
    
    features = load_features(SCHEMA_PATH)
    print(f"特征数: {len(features)}")
    
    df = load_data(DATA_PATH, SAMPLE_DATE, features, NUM_FILES)
    
    cov_df = analyze_coverage(df, features)
    lift_df = analyze_lift(df, features)
    red = analyze_redundancy(features)
    
    result = final_score(cov_df, lift_df, red)
    
    # 保存
    cov_df.to_csv(f"{OUTPUT_DIR}/coverage.csv", index=False)
    lift_df.to_csv(f"{OUTPUT_DIR}/lift.csv", index=False)
    result.to_csv(f"{OUTPUT_DIR}/importance.csv", index=False)
    
    with open(f"{OUTPUT_DIR}/redundancy.json", 'w') as f:
        json.dump(red, f, indent=2, default=str)
    
    # 统计
    print("\n" + "=" * 50)
    print("结果")
    print("=" * 50)
    print(result['action'].value_counts())
    print(f"\n状态分布:")
    print(result['status'].value_counts())
    
    remove = result[result['action'] == 'REMOVE']['feature'].tolist()
    with open(f"{OUTPUT_DIR}/to_remove.txt", 'w') as f:
        f.write('\n'.join(remove))
    
    print(f"\n建议删除: {len(remove)} 个")
    print(f"输出: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
