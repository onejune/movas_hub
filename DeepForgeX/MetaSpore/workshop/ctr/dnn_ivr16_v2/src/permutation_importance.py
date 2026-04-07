#!/usr/bin/env python3
"""
Permutation Importance 分析（基于保存的预测结果）

方法: 通过分析特征值分布与预测值的关系来评估特征重要性
- 对分类特征: 计算每个特征值对应的预测均值方差
- 对数值特征: 计算与预测值的相关系数

注意: 这不是真正的 permutation importance（需要重新推理），
但可以作为特征重要性的近似估计。
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
import glob

# 配置
PRED_PATH = "./output/predictions_2026-03-03"
SCHEMA_PATH = "./conf/combine_schema"
OUTPUT_DIR = "./output/permutation_importance"
DATA_PATH = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet/2026-03-03"


def load_features(schema_path):
    """加载特征列表"""
    features = []
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def load_predictions(pred_path, num_files=50):
    """加载预测结果"""
    files = sorted(glob.glob(f"{pred_path}/part-*.parquet"))[:num_files]
    print(f"加载 {len(files)} 个预测文件...")
    
    dfs = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"预测数据: {len(df)} 行, 列: {df.columns.tolist()}")
    return df


def load_feature_data(data_path, features, num_files=10):
    """加载原始特征数据"""
    files = sorted(glob.glob(f"{data_path}/part-*.parquet"))[:num_files]
    print(f"加载 {len(files)} 个特征文件...")
    
    # 只加载需要的特征
    base_features = list(set([f.split('#')[0] for f in features]))
    
    dfs = []
    for f in files:
        pf = pq.ParquetFile(f)
        available = set(pf.schema.names)
        cols = [c for c in base_features if c in available]
        df = pq.read_table(f, columns=cols).to_pandas()
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"特征数据: {len(df)} 行, {len(df.columns)} 列")
    return df


def analyze_feature_importance(pred_df, feature_df, features):
    """分析特征重要性"""
    print("\n=== 特征重要性分析 ===")
    
    # 获取预测列
    pred_col = 'rawPrediction' if 'rawPrediction' in pred_df.columns else 'prediction'
    y_pred = pred_df[pred_col].values
    y_true = pred_df['label'].values
    
    # 对齐数据
    min_len = min(len(y_pred), len(feature_df))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    feature_df = feature_df.iloc[:min_len].reset_index(drop=True)
    
    base_auc = roc_auc_score(y_true, y_pred)
    print(f"基准 AUC: {base_auc:.4f}")
    print(f"样本数: {min_len}")
    
    results = []
    
    for i, feat in enumerate(features):
        base_feat = feat.split('#')[0]
        
        if base_feat not in feature_df.columns:
            results.append({
                'feature': feat,
                'importance': 0,
                'method': 'NOT_IN_DATA',
                'unique_values': 0,
                'coverage': 0,
            })
            continue
        
        col = feature_df[base_feat]
        non_null = col.notna().sum() / len(col)
        unique = col.nunique()
        
        # 根据特征类型选择分析方法
        if col.dtype in ['object', 'category'] or unique < 100:
            # 分类特征: 计算每个值的预测均值方差
            try:
                grp = pd.DataFrame({'feat': col.fillna('__NULL__'), 'pred': y_pred})
                grp = grp.groupby('feat')['pred'].agg(['mean', 'count'])
                grp = grp[grp['count'] >= 50]  # 至少50个样本
                
                if len(grp) > 1:
                    importance = grp['mean'].var()
                    method = 'categorical_variance'
                else:
                    importance = 0
                    method = 'single_value'
            except Exception as e:
                importance = 0
                method = f'error: {e}'
        else:
            # 数值特征: 计算相关系数
            try:
                valid_mask = col.notna()
                if valid_mask.sum() > 100:
                    corr = np.abs(np.corrcoef(col[valid_mask].astype(float), y_pred[valid_mask])[0, 1])
                    importance = corr if not np.isnan(corr) else 0
                    method = 'correlation'
                else:
                    importance = 0
                    method = 'insufficient_data'
            except Exception as e:
                importance = 0
                method = f'error: {e}'
        
        results.append({
            'feature': feat,
            'importance': round(float(importance), 6),
            'method': method,
            'unique_values': int(unique),
            'coverage': round(float(non_null), 4),
        })
        
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(features)}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('importance', ascending=False)
    
    # 分类
    valid_df = df[df['importance'] > 0]
    if len(valid_df) > 0:
        p75 = valid_df['importance'].quantile(0.75)
        p50 = valid_df['importance'].quantile(0.50)
        p25 = valid_df['importance'].quantile(0.25)
        
        def classify(row):
            if row['importance'] == 0:
                return 'NO_SIGNAL'
            imp = row['importance']
            if imp >= p75:
                return 'HIGH'
            elif imp >= p50:
                return 'MEDIUM_HIGH'
            elif imp >= p25:
                return 'MEDIUM_LOW'
            else:
                return 'LOW'
        
        df['importance_level'] = df.apply(classify, axis=1)
    else:
        df['importance_level'] = 'NO_SIGNAL'
    
    df['rank'] = range(1, len(df) + 1)
    
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Permutation Importance 分析")
    print("=" * 60)
    
    # 加载特征列表
    features = load_features(SCHEMA_PATH)
    print(f"特征数: {len(features)}")
    
    # 加载预测结果
    pred_df = load_predictions(PRED_PATH, num_files=50)
    
    # 加载特征数据
    feature_df = load_feature_data(DATA_PATH, features, num_files=10)
    
    # 分析
    result = analyze_feature_importance(pred_df, feature_df, features)
    
    # 保存结果
    result.to_csv(os.path.join(OUTPUT_DIR, "importance.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("Top 30 重要特征")
    print("=" * 60)
    print(result.head(30).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Bottom 20 特征")
    print("=" * 60)
    print(result.tail(20).to_string(index=False))
    
    print("\n--- 重要性分布 ---")
    print(result['importance_level'].value_counts())
    
    # 保存建议删除的特征
    to_remove = result[result['importance_level'].isin(['NO_SIGNAL', 'LOW'])]['feature'].tolist()
    with open(os.path.join(OUTPUT_DIR, "to_remove.txt"), 'w') as f:
        for feat in to_remove:
            f.write(feat + '\n')
    print(f"\n建议删除的特征: {len(to_remove)} 个")
    
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
