#!/usr/bin/env python3
"""
特征打乱模块

用于 Permutation Importance 分析，在数据读取后打乱指定特征的值。
"""

import os
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def shuffle_feature_column(df: DataFrame, feature_name: str, seed: int = 42) -> DataFrame:
    """
    打乱 DataFrame 中指定特征列的值
    
    Args:
        df: Spark DataFrame
        feature_name: 要打乱的特征名
        seed: 随机种子
    
    Returns:
        打乱后的 DataFrame
    """
    if feature_name not in df.columns:
        print(f"警告: 特征 {feature_name} 不在 DataFrame 中，跳过打乱")
        return df
    
    # 方法: 添加随机排序列，按随机顺序重新分配特征值
    # 1. 给每行添加原始行号
    df = df.withColumn("_row_id", F.monotonically_increasing_id())
    
    # 2. 提取特征列并随机排序
    feature_df = df.select("_row_id", feature_name)
    feature_df = feature_df.withColumn("_rand", F.rand(seed))
    
    # 3. 创建新的行号映射（随机打乱）
    window = Window.orderBy("_rand")
    feature_df = feature_df.withColumn("_new_row_id", F.row_number().over(window) - 1)
    
    # 4. 获取原始行号的映射
    original_ids = df.select("_row_id").withColumn("_orig_order", F.row_number().over(Window.orderBy("_row_id")) - 1)
    
    # 5. 通过 join 重新分配特征值
    shuffled_feature = feature_df.select(
        F.col("_new_row_id").alias("_orig_order"),
        F.col(feature_name).alias(f"{feature_name}_shuffled")
    )
    
    # 6. Join 回原始 DataFrame
    df = df.join(original_ids.join(shuffled_feature, on="_orig_order"), on="_row_id")
    
    # 7. 替换原特征列
    df = df.drop(feature_name).withColumnRenamed(f"{feature_name}_shuffled", feature_name)
    
    # 8. 清理临时列
    df = df.drop("_row_id", "_orig_order")
    
    return df


def shuffle_feature_simple(df: DataFrame, feature_name: str, seed: int = 42) -> DataFrame:
    """
    简化版特征打乱（更高效，但不保证完全随机）
    
    通过对特征值进行随机采样替换来实现打乱效果
    """
    if feature_name not in df.columns:
        print(f"警告: 特征 {feature_name} 不在 DataFrame 中，跳过打乱")
        return df
    
    # 获取特征的所有唯一值
    distinct_values = df.select(feature_name).distinct().collect()
    values = [row[feature_name] for row in distinct_values]
    
    if len(values) <= 1:
        print(f"警告: 特征 {feature_name} 只有 {len(values)} 个唯一值，打乱无意义")
        return df
    
    # 使用 Spark 的 shuffle 函数
    # 添加随机数列，然后按随机数排序后取特征值
    df = df.withColumn("_rand", F.rand(seed))
    
    # 创建打乱后的特征值（通过 row_number 和 模运算实现伪随机重分配）
    window = Window.orderBy("_rand")
    
    # 获取原始特征值的排序
    feature_window = Window.partitionBy(F.lit(1)).orderBy(feature_name)
    
    df = df.withColumn(
        f"{feature_name}_shuffled",
        F.when(
            F.rand(seed + 1) > 0.5,
            F.lead(feature_name, 1).over(Window.orderBy("_rand"))
        ).otherwise(
            F.lag(feature_name, 1).over(Window.orderBy("_rand"))
        )
    )
    
    # 处理边界情况（首尾行）
    df = df.withColumn(
        feature_name,
        F.coalesce(F.col(f"{feature_name}_shuffled"), F.col(feature_name))
    )
    
    df = df.drop("_rand", f"{feature_name}_shuffled")
    
    return df


def get_shuffle_feature_from_env() -> str:
    """从环境变量获取要打乱的特征名"""
    return os.environ.get("SHUFFLE_FEATURE", None)


def apply_feature_shuffle(df: DataFrame, feature_name: str = None) -> DataFrame:
    """
    应用特征打乱（如果指定了特征）
    
    Args:
        df: Spark DataFrame
        feature_name: 要打乱的特征名，如果为 None 则从环境变量读取
    
    Returns:
        处理后的 DataFrame
    """
    if feature_name is None:
        feature_name = get_shuffle_feature_from_env()
    
    if feature_name:
        print(f"[Permutation Importance] 打乱特征: {feature_name}")
        df = shuffle_feature_column(df, feature_name)
    
    return df
