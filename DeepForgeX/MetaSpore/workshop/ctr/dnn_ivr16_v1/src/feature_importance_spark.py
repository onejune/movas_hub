#!/usr/bin/env python3
"""
特征重要性评估 - 本地 PySpark 版本
使用全量数据，更可靠的统计
"""

import os
import json
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# 配置
DATA_PATH = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet"
SCHEMA_PATH = "./conf/combine_schema"
OUTPUT_DIR = "./output/feature_importance"
SAMPLE_DATE = "2026-03-03"


def init_spark():
    """本地模式 Spark"""
    spark = SparkSession.builder \
        .appName("FeatureImportanceLocal") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.local.dir", "/tmp/spark-local") \
        .config("spark.sql.parquet.enableVectorizedReader", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_features(schema_path):
    """加载特征列表"""
    features = []
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def analyze_all(spark, data_path, date, features):
    """一次性计算所有特征的覆盖率和 lift"""
    print(f"\n读取数据: {data_path}/{date}")
    
    df = spark.read.parquet(f"{data_path}/{date}")
    total = df.count()
    print(f"总样本数: {total:,}")
    
    # 整体正样本率
    pos_rate = df.agg(F.avg("label")).collect()[0][0]
    print(f"正样本率: {pos_rate:.4f}")
    
    # 提取基础特征名
    base_features = list(set([f.split('#')[0] for f in features]))
    available_cols = set(df.columns)
    base_features = [f for f in base_features if f in available_cols]
    print(f"可用特征: {len(base_features)}/{len(features)}")
    
    results = []
    
    for i, feat in enumerate(base_features):
        try:
            # 覆盖率
            col = F.col(feat)
            
            # 判断类型
            dtype = dict(df.dtypes).get(feat, 'string')
            
            if dtype == 'string':
                non_null_count = df.filter(
                    col.isNotNull() & (col != '') & (col != 'null') & (col != 'None')
                ).count()
            else:
                non_null_count = df.filter(col.isNotNull()).count()
            
            coverage = non_null_count / total
            
            # 基数
            cardinality = df.select(feat).distinct().count()
            
            # Lift 分析
            grouped = df.groupBy(feat).agg(
                F.avg("label").alias("pos_rate"),
                F.count("*").alias("cnt")
            ).filter(F.col("cnt") >= 100).collect()
            
            if len(grouped) >= 2:
                rates = [r["pos_rate"] for r in grouped if r["pos_rate"] is not None]
                max_lift = max(rates) / pos_rate if pos_rate > 0 and rates else 1
                min_lift = min(rates) / pos_rate if pos_rate > 0 and rates else 1
                lift_range = max_lift - min_lift
            else:
                max_lift, min_lift, lift_range = 1, 1, 0
            
            # 状态判断
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
                'cardinality': cardinality,
                'status': status,
                'max_lift': round(max_lift, 4),
                'min_lift': round(min_lift, 4),
                'lift_range': round(lift_range, 4)
            })
            
        except Exception as e:
            print(f"  特征 {feat} 失败: {e}")
            results.append({
                'feature': feat,
                'coverage': 0,
                'cardinality': 0,
                'status': 'ERROR',
                'max_lift': 0,
                'min_lift': 0,
                'lift_range': 0
            })
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(base_features)}")
    
    return results, pos_rate


def analyze_redundancy(features):
    """时间窗口冗余分析"""
    print("\n=== 冗余分析 ===")
    
    time_windows = ['1d', '3d', '7d', '14d', '15d', '30d', '60d', '90d', '180d',
                    '1h', '3h', '12h', '24h', '48h', '3m', '10m', '30m',
                    '1_3d', '4_10d', '11_30d', '31_60d', '61_90d', '61_180d']
    
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
    return {'groups': {k: [f for f, _ in v] for k, v in groups.items()}, 'redundant': redundant}


def calculate_score(results, redundancy):
    """综合评分"""
    redundant_set = set(redundancy.get('redundant', []))
    
    for r in results:
        feat = r['feature']
        cov = r.get('coverage', 0)
        lift = r.get('lift_range', 0)
        
        # 覆盖率得分
        cov_s = 100 if cov >= 0.5 else (70 if cov >= 0.1 else (40 if cov >= 0.01 else 10))
        
        # Lift 得分
        lift_s = 100 if lift >= 2 else (80 if lift >= 1 else (60 if lift >= 0.5 else (40 if lift >= 0.1 else 20)))
        
        # 冗余惩罚
        is_redundant = feat in redundant_set or any(feat.startswith(rf.split('#')[0]) for rf in redundant_set)
        penalty = -30 if is_redundant else 0
        
        score = max(0, min(100, cov_s * 0.3 + lift_s * 0.5 + penalty))
        
        r['is_redundant'] = is_redundant
        r['score'] = score
        r['action'] = 'KEEP' if score >= 55 else ('REVIEW' if score >= 35 else 'REMOVE')
    
    return sorted(results, key=lambda x: x['score'], reverse=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("特征重要性评估 (本地 PySpark)")
    print("=" * 60)
    
    spark = init_spark()
    features = load_features(SCHEMA_PATH)
    print(f"特征数: {len(features)}")
    
    # 分析
    results, pos_rate = analyze_all(spark, DATA_PATH, SAMPLE_DATE, features)
    redundancy = analyze_redundancy(features)
    final_results = calculate_score(results, redundancy)
    
    # 保存 CSV
    import csv
    with open(f"{OUTPUT_DIR}/importance_full.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature', 'coverage', 'cardinality', 'status', 
                                                'max_lift', 'min_lift', 'lift_range', 
                                                'is_redundant', 'score', 'action'])
        writer.writeheader()
        writer.writerows(final_results)
    
    # 保存冗余分析
    with open(f"{OUTPUT_DIR}/redundancy.json", 'w') as f:
        json.dump(redundancy, f, indent=2, ensure_ascii=False)
    
    # 统计
    print("\n" + "=" * 60)
    print("结果统计")
    print("=" * 60)
    
    from collections import Counter
    action_counts = Counter(r['action'] for r in final_results)
    status_counts = Counter(r['status'] for r in final_results)
    
    print(f"\n推荐动作:")
    for action, cnt in action_counts.most_common():
        print(f"  {action}: {cnt}")
    
    print(f"\n状态分布:")
    for status, cnt in status_counts.most_common():
        print(f"  {status}: {cnt}")
    
    # 建议删除列表
    remove_list = [r['feature'] for r in final_results if r['action'] == 'REMOVE']
    with open(f"{OUTPUT_DIR}/to_remove_full.txt", 'w') as f:
        f.write('\n'.join(remove_list))
    
    print(f"\n建议删除: {len(remove_list)} 个特征")
    print(f"输出目录: {OUTPUT_DIR}/")
    
    # Top 20 特征
    print(f"\n=== Top 20 高价值特征 ===")
    for r in final_results[:20]:
        print(f"  {r['feature']}: lift={r['lift_range']:.2f}, cov={r['coverage']:.2f}, score={r['score']:.0f}")
    
    spark.stop()


if __name__ == "__main__":
    main()
