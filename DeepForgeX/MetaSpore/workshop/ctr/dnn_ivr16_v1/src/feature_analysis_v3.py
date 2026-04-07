#!/usr/bin/env python3
"""
特征重要性分析脚本 V3 - 修复版本
正确处理字符串和数值特征的覆盖率计算
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from collections import defaultdict

# 配置
DATA_PATH = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet"
SAMPLE_DATE = "2026-03-20"
COMBINE_SCHEMA_PATH = "./conf/combine_schema"
OUTPUT_PATH = "./output/feature_analysis"

def init_spark():
    spark = SparkSession.builder \
        .appName("FeatureAnalysisV3") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_combine_schema(path):
    features = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if '#' in line:
                    feat_name = line.split('#')[0]
                else:
                    feat_name = line
                features.append((line, feat_name))
    return features

def is_numeric_column(df, col_name):
    """检查列是否为数值类型"""
    dtype = dict(df.dtypes).get(col_name, 'string')
    return dtype in ['int', 'bigint', 'long', 'float', 'double', 'decimal']

def analyze_features_batch(spark, df, features, label_col='label'):
    results = []
    
    total_count = df.count()
    label_stats = df.groupBy(label_col).count().collect()
    label_counts = {row[label_col]: row['count'] for row in label_stats}
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    
    print(f"Total samples: {total_count:,}, Pos: {pos_count:,}, Neg: {neg_count:,}")
    print(f"Positive rate: {pos_count/total_count*100:.2f}%")
    
    all_columns = set(df.columns)
    col_types = dict(df.dtypes)
    
    print("\nCalculating feature statistics...")
    
    # 分批处理
    batch_size = 30
    valid_features = [(s, f) for s, f in features if f in all_columns]
    missing_features = [(s, f) for s, f in features if f not in all_columns]
    
    # 先处理缺失特征
    for schema_name, feat_name in missing_features:
        results.append({
            'schema_name': schema_name,
            'feature_name': feat_name,
            'exists': False,
            'coverage': 0,
            'null_ratio': 1.0,
            'distinct_count': 0,
            'pos_coverage': 0,
            'neg_coverage': 0,
            'coverage_lift': 0,
        })
    
    print(f"Missing features: {len(missing_features)}")
    print(f"Valid features: {len(valid_features)}")
    
    for i in range(0, len(valid_features), batch_size):
        batch = valid_features[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(valid_features)-1)//batch_size + 1}...")
        
        agg_exprs = []
        for schema_name, feat_name in batch:
            is_numeric = col_types.get(feat_name, 'string') in ['int', 'bigint', 'long', 'float', 'double', 'decimal']
            
            # 对于数值特征：非空且非0视为有效
            # 对于字符串特征：非空且非空字符串且非'null'字符串视为有效
            if is_numeric:
                valid_cond = F.col(feat_name).isNotNull() & (F.col(feat_name) != 0)
            else:
                valid_cond = F.col(feat_name).isNotNull() & (F.col(feat_name) != '') & (F.col(feat_name) != 'null') & (F.col(feat_name) != '-')
            
            # 总体有效计数
            agg_exprs.append(F.count(F.when(valid_cond, 1)).alias(f'{feat_name}__valid'))
            # 正样本有效计数
            agg_exprs.append(F.count(F.when(valid_cond & (F.col(label_col) == 1), 1)).alias(f'{feat_name}__pos_valid'))
            # distinct count
            agg_exprs.append(F.countDistinct(feat_name).alias(f'{feat_name}__distinct'))
        
        batch_stats = df.agg(*agg_exprs).collect()[0].asDict()
        
        for schema_name, feat_name in batch:
            valid_count = batch_stats.get(f'{feat_name}__valid', 0)
            pos_valid = batch_stats.get(f'{feat_name}__pos_valid', 0)
            distinct_count = batch_stats.get(f'{feat_name}__distinct', 0)
            
            coverage = valid_count / total_count if total_count > 0 else 0
            pos_coverage = pos_valid / pos_count if pos_count > 0 else 0
            neg_valid = valid_count - pos_valid
            neg_coverage = neg_valid / neg_count if neg_count > 0 else 0
            coverage_lift = pos_coverage / neg_coverage if neg_coverage > 0 else 0
            
            results.append({
                'schema_name': schema_name,
                'feature_name': feat_name,
                'exists': True,
                'coverage': coverage,
                'null_ratio': 1 - coverage,
                'distinct_count': distinct_count,
                'pos_coverage': pos_coverage,
                'neg_coverage': neg_coverage,
                'coverage_lift': coverage_lift,
            })
    
    return pd.DataFrame(results)

def categorize_features(df_results):
    df = df_results.copy()
    
    def score_feature(row):
        score = 0
        
        # 覆盖率得分 (0-30分)
        if row['coverage'] >= 0.8:
            score += 30
        elif row['coverage'] >= 0.5:
            score += 25
        elif row['coverage'] >= 0.2:
            score += 20
        elif row['coverage'] >= 0.1:
            score += 15
        elif row['coverage'] >= 0.05:
            score += 10
        elif row['coverage'] >= 0.01:
            score += 5
        
        # 区分度得分 (0-40分)
        lift = row['coverage_lift']
        if lift > 3.0 or lift < 0.33:
            score += 40
        elif lift > 2.0 or lift < 0.5:
            score += 35
        elif lift > 1.5 or lift < 0.67:
            score += 25
        elif lift > 1.2 or lift < 0.83:
            score += 15
        elif lift > 1.1 or lift < 0.91:
            score += 5
        
        # distinct_count得分 (0-20分)
        distinct = row['distinct_count']
        if 10 <= distinct <= 10000:
            score += 20
        elif 3 <= distinct < 10 or 10000 < distinct <= 100000:
            score += 15
        elif distinct > 100000:
            score += 10
        elif distinct == 2:
            score += 8
        
        # 惩罚项
        if row['coverage'] < 0.001:
            score -= 30
        
        return max(0, score)
    
    df['importance_score'] = df.apply(score_feature, axis=1)
    
    def categorize(row):
        if not row['exists']:
            return 'NOT_IN_DATA'
        if row['coverage'] < 0.0001:
            return 'EXTREMELY_SPARSE'
        if row['coverage'] < 0.001:
            return 'VERY_SPARSE'
        if row['coverage'] < 0.01:
            return 'SPARSE'
        if row['importance_score'] >= 55:
            return 'HIGH_VALUE'
        if row['importance_score'] >= 40:
            return 'MEDIUM_VALUE'
        if row['importance_score'] >= 25:
            return 'LOW_VALUE'
        return 'CANDIDATE_REMOVE'
    
    df['category'] = df.apply(categorize, axis=1)
    
    return df.sort_values('importance_score', ascending=False)

def identify_redundant_features(df_results):
    """识别冗余特征 - 同一前缀的多个时间窗口"""
    redundant = []
    
    feature_groups = defaultdict(list)
    for _, row in df_results.iterrows():
        if not row['exists'] or row['coverage'] < 0.001:
            continue
            
        feat = row['feature_name']
        # 提取特征前缀
        for suffix in ['_180d', '_90d', '_60d', '_30d', '_15d', '_14d', '_7d', '_3d', '_1d',
                       '_48h', '_24h', '_12h', '_3h', '_1h', '_30m', '_10m', '_3m',
                       '_61_180d', '_31_60d', '_11_30d', '_4_10d', '_1_3d']:
            if feat.endswith(suffix):
                prefix = feat[:-len(suffix)]
                feature_groups[prefix].append((feat, row['importance_score'], row['coverage'], suffix))
                break
    
    # 对于同一组特征，保留得分最高的3个
    for prefix, feats in feature_groups.items():
        if len(feats) > 3:
            sorted_feats = sorted(feats, key=lambda x: x[1], reverse=True)
            for feat, score, coverage, suffix in sorted_feats[3:]:
                redundant.append((feat, f"Redundant window ({suffix}), keep top 3 of {prefix}"))
    
    return redundant

def main():
    print("=" * 60)
    print("Feature Analysis V3 for dnn_ivr16_v1")
    print("=" * 60)
    
    spark = init_spark()
    
    data_path = f"{DATA_PATH}/{SAMPLE_DATE}"
    print(f"\nLoading data from: {data_path}")
    df = spark.read.parquet(data_path)
    
    features = load_combine_schema(COMBINE_SCHEMA_PATH)
    print(f"Loaded {len(features)} features from combine_schema")
    
    print("\n" + "=" * 60)
    print("Analyzing feature coverage and importance...")
    print("=" * 60)
    df_results = analyze_features_batch(spark, df, features)
    
    df_results = categorize_features(df_results)
    
    redundant_features = identify_redundant_features(df_results)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = f"{OUTPUT_PATH}/feature_analysis_result_v3.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Category Distribution:")
    print(df_results['category'].value_counts().to_string())
    
    # 高价值特征
    high_value = df_results[df_results['category'] == 'HIGH_VALUE']
    print(f"\n✅ HIGH VALUE Features ({len(high_value)}):")
    for _, row in high_value.head(40).iterrows():
        print(f"  - {row['schema_name']}: score={row['importance_score']}, "
              f"coverage={row['coverage']:.4f}, lift={row['coverage_lift']:.2f}, distinct={row['distinct_count']}")
    
    # 中等价值
    medium_value = df_results[df_results['category'] == 'MEDIUM_VALUE']
    print(f"\n⚠️ MEDIUM VALUE Features ({len(medium_value)}):")
    for _, row in medium_value.head(20).iterrows():
        print(f"  - {row['schema_name']}: score={row['importance_score']}, coverage={row['coverage']:.4f}")
    
    # 低价值/稀疏特征
    low_sparse = df_results[df_results['category'].isin(['LOW_VALUE', 'SPARSE', 'VERY_SPARSE', 'EXTREMELY_SPARSE', 'CANDIDATE_REMOVE'])]
    print(f"\n🔻 LOW VALUE / SPARSE Features ({len(low_sparse)}):")
    for _, row in low_sparse.head(30).iterrows():
        print(f"  - {row['schema_name']}: category={row['category']}, coverage={row['coverage']:.6f}")
    
    # 冗余特征
    print(f"\n🔄 REDUNDANT FEATURES ({len(redundant_features)}):")
    for feat, reason in redundant_features[:40]:
        print(f"  - {feat}: {reason}")
    
    # 生成优化后的特征列表
    print("\n" + "=" * 60)
    print("GENERATING OPTIMIZED FEATURE LIST")
    print("=" * 60)
    
    redundant_set = set([f[0] for f in redundant_features])
    keep_features = []
    remove_features = []
    
    for _, row in df_results.iterrows():
        feat = row['schema_name']
        feat_name = row['feature_name']
        
        if not row['exists']:
            remove_features.append((feat, 'NOT_IN_DATA'))
        elif row['category'] in ['EXTREMELY_SPARSE', 'VERY_SPARSE']:
            remove_features.append((feat, row['category']))
        elif row['category'] == 'CANDIDATE_REMOVE':
            remove_features.append((feat, 'LOW_SCORE'))
        elif feat_name in redundant_set:
            remove_features.append((feat, 'REDUNDANT'))
        else:
            keep_features.append(feat)
    
    print(f"\n📋 Keep: {len(keep_features)} features")
    print(f"📋 Remove: {len(remove_features)} features")
    
    # 保存优化后的combine_schema
    optimized_schema_path = f"{OUTPUT_PATH}/combine_schema_optimized_v3"
    with open(optimized_schema_path, 'w') as f:
        for feat in keep_features:
            f.write(feat + '\n')
    print(f"\nOptimized schema saved to: {optimized_schema_path}")
    
    # 保存删除列表
    remove_list_path = f"{OUTPUT_PATH}/features_to_remove_v3.txt"
    with open(remove_list_path, 'w') as f:
        f.write("# Features recommended for removal\n")
        f.write("# Format: feature_name | reason\n\n")
        for feat, reason in remove_features:
            f.write(f"{feat} | {reason}\n")
    print(f"Remove list saved to: {remove_list_path}")
    
    # 建议的交叉特征
    cross_features = [
        'country#demand_pkgname',
        'business_type#devicetype', 
        'adx#bundle',
        'country#business_type',
        'os#demand_pkgname',
        'country#adx',
    ]
    
    print(f"\n➕ SUGGESTED CROSS FEATURES TO ADD:")
    for cf in cross_features:
        print(f"  - {cf}")
    
    # 保存带交叉特征的schema
    cross_schema_path = f"{OUTPUT_PATH}/combine_schema_with_cross_v3"
    with open(cross_schema_path, 'w') as f:
        for feat in keep_features:
            f.write(feat + '\n')
        f.write("\n# Cross features\n")
        for cf in cross_features:
            f.write(cf + '\n')
    print(f"\nSchema with cross features saved to: {cross_schema_path}")
    
    spark.stop()
    print("\n✅ Analysis complete!")
    return df_results

if __name__ == "__main__":
    main()
