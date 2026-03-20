#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WinAdapt TrainFlow - 基于 MetaSpore 的 DEFER 训练流程

完全对齐 src_tf_github 实现:
- WinAdaptDNN: 4 输出头 (cv, time_w1, time_w2, time_w3)
- delay_win_select_loss: 14 维标签损失
- 时间窗口: 24h, 48h, 72h
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Spark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ArrayType, StringType

# PyTorch
import torch
import torch.nn as nn

# MetaSpore - 使用 DeepForgeX 目录
sys.path.insert(0, "/mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/MetaSpore/python")

from metaspore.algos.delay_feedback.defer_models import WinAdaptDNN, DeferDNN, create_defer_model
from metaspore.algos.delay_feedback.defer_loss import delay_win_select_loss

# 基类 - 使用 ms_dnn_wd.py 中的 MsModelTrainFlow (defer_v1 成功时用的版本)
from ms_dnn_wd import MsModelTrainFlow
from movas_logger import MovasLogger


# ============================================================================
# 常量
# ============================================================================

# 时间窗口 (小时)
WIN1_HOURS = 24.0   # 窗口1
WIN2_HOURS = 48.0   # 窗口2
WIN3_HOURS = 72.0   # 窗口3
DELAY_WINDOW_HOURS = 168.0  # 归因窗口 7 天

# 过滤业务类型
FILTER_BUSINESS_TYPES = ["aecps", "shein", "lazada_rta", "lazada_cps", "aedsp", "aerta"]


# ============================================================================
# WinAdapt 标签构建
# ============================================================================

def build_winadapt_labels_udf(win1=WIN1_HOURS, win2=WIN2_HOURS, win3=WIN3_HOURS):
    """
    构建 WinAdapt 14 维标签的 UDF
    
    输入: label (int), diff_hours (float)
    输出: 14 维 float 数组
    """
    @F.udf(ArrayType(FloatType()))
    def _build_labels(label: int, diff_hours: float) -> List[float]:
        import math
        
        if diff_hours is None:
            diff_hours = 9999.0
        else:
            try:
                diff_hours = float(diff_hours)
            except (ValueError, TypeError):
                diff_hours = 9999.0
            if isinstance(diff_hours, float) and math.isnan(diff_hours):
                diff_hours = 9999.0
        if diff_hours >= 2400:
            diff_hours = 9999.0
        
        labels = [0.0] * 14
        
        is_positive = (label == 1)
        is_negative = (label == 0)
        
        # 窗口判断
        in_win1 = is_positive and (0 < diff_hours <= win1)
        in_win2 = is_positive and (win1 < diff_hours <= win2)
        in_win3 = is_positive and (win2 < diff_hours <= win3)
        delayed = is_positive and (diff_hours > win3)
        
        # 构建标签
        labels[0] = 1.0 if delayed else 0.0  # label_11 (delayed positive)
        labels[1] = 1.0 if is_negative else 0.0  # label_10 (true negative)
        labels[2] = 1.0 if in_win1 else 0.0  # label_01_15 (win1 positive)
        labels[3] = 1.0 if in_win2 else 0.0  # label_01_30 (win2 positive, incremental)
        labels[4] = 1.0 if in_win3 else 0.0  # label_01_60 (win3 positive, incremental)
        labels[5] = 1.0 if (in_win1 or in_win2) else 0.0  # label_01_30_sum (win1+win2)
        labels[6] = 1.0 if (in_win1 or in_win2 or in_win3) else 0.0  # label_01_60_sum (win1+win2+win3)
        labels[7] = 1.0  # label_01_30_mask
        labels[8] = 1.0  # label_01_60_mask
        labels[9] = 0.0  # label_00 (unobserved negative, not used)
        labels[10] = 1.0 if (in_win1 or in_win2 or in_win3) else 0.0  # label_01 (all window positive)
        
        # label_11_xx: delayed positive breakdown
        if delayed:
            if diff_hours <= win3 * 2:
                labels[11] = 1.0  # label_11_15
            elif diff_hours <= win3 * 3:
                labels[12] = 1.0  # label_11_30
            else:
                labels[13] = 1.0  # label_11_60
        
        return labels
    
    return _build_labels


# ============================================================================
# DeferTrainFlow
# ============================================================================

class DeferTrainFlow(MsModelTrainFlow):
    """
    DEFER (WinAdapt) 训练流程
    
    继承 MsModelTrainFlow，重写关键方法:
    - _read_dataset_by_date: 使用 sample_date 分区格式
    - _build_model_module: 构建 WinAdaptDNN 模型
    - _train_model: 添加 DEFER 标签构建
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # 时间窗口配置 (从 params 读取)
        defer_windows = self.params.get("defer_windows", [WIN1_HOURS, WIN2_HOURS, WIN3_HOURS])
        self.win1_hours = float(defer_windows[0]) if len(defer_windows) > 0 else WIN1_HOURS
        self.win2_hours = float(defer_windows[1]) if len(defer_windows) > 1 else WIN2_HOURS
        self.win3_hours = float(defer_windows[2]) if len(defer_windows) > 2 else WIN3_HOURS
        
        print(f"DEFER windows: {self.win1_hours}h, {self.win2_hours}h, {self.win3_hours}h")
    
    def _read_dataset_by_date(self, base_path, date_str):
        """
        重写数据读取方法 - 自动检测分区格式 (part= 或 sample_date=)
        """
        # 检测分区格式
        part_path = os.path.join(base_path, f"part={date_str}")
        sample_date_path = os.path.join(base_path, f"sample_date={date_str}")
        
        if os.path.exists(part_path):
            data_path = part_path
        elif os.path.exists(sample_date_path):
            data_path = sample_date_path
        else:
            raise FileNotFoundError(f"No data found for date {date_str} in {base_path}")
        MovasLogger.add_log(content=f"Reading DEFER Parquet from: {data_path}")
        
        df = self.spark_session.read.parquet(data_path)
        
        # 保留 diff_hours 用于标签构建
        used_cols = list(self.used_fea_list) + ['diff_hours']
        used_cols = [c for c in used_cols if c in df.columns]
        df = df.select(*used_cols)
        
        # 类型转换 (保持 diff_hours 为 float)
        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            elif col_name == 'diff_hours':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        
        df = self.random_sample(df)
        df = df.fillna('unknown')
        
        MovasLogger.add_log(content=f"DEFER Parquet loaded, columns: {len(df.columns)}")
        return df
    
    def _build_model_module(self):
        """
        构建 DEFER 模型
        
        根据配置 model_type 选择:
        - WinAdaptDNN (默认): 单输出层，适合长窗口 (24/48/72h)
        - DeferDNN: 多输出头，适合短窗口 (15/30/60min)
        """
        model_type = self.params.get("model_type", "WinAdaptDNN")
        MovasLogger.add_log(content=f"Building DEFER model: {model_type}")
        
        model_kwargs = dict(
            embedding_dim=self.embedding_size,
            column_name_path=self.combine_schema_path,
            combine_schema_path=self.combine_schema_path,
            dnn_hidden_units=self.dnn_hidden_units,
            dnn_hidden_activations="ReLU",
            net_dropout=self.net_dropout,
            batch_norm=self.batch_norm,
            use_bias=True,
            adam_learning_rate=self.adam_learning_rate,
            ftrl_l1=self.ftrl_l1,
            ftrl_l2=self.ftrl_l2,
            ftrl_alpha=self.ftrl_alpha,
            ftrl_beta=self.ftrl_beta,
        )
        
        self.model_module = create_defer_model(model_type, **model_kwargs)
    
    def _build_defer_labels(self, df: DataFrame) -> DataFrame:
        """
        构建 DEFER 14 维标签 - 使用纯 Spark SQL 表达式，避免 Python UDF
        
        背景:
        -----
        DEFER (Delayed Feedback Modeling) 处理延迟转化问题。用户点击广告后，
        转化可能在几分钟到几天后发生。传统方法将"未转化"样本直接标为负样本，
        但这些样本可能只是"尚未观测到转化"，造成标签噪声。
        
        DEFER 将样本分为 4 类:
        - 窗口内正样本 (label_01): 在观测窗口内转化
        - 延迟正样本 (label_11): 超过窗口但最终转化
        - 真负样本 (label_10): 超过归因窗口仍未转化
        - 未观测样本 (label_00): 窗口内未转化，但归因窗口未结束 (不用于训练)
        
        时间窗口示例 (v2 配置):
        ----------------------
        假设 win1=24h, win2=48h, win3=72h, delay=168h (7天):
        
        点击时刻 ──┬── 24h ──┬── 48h ──┬── 72h ──┬─────── 168h ────┬───→ 时间
                   │  win1   │  win2   │  win3   │   delay window   │
                   │         │         │         │                  │
        转化在此 → label_01_15 (窗口1正样本)
                             → label_01_30 (窗口2正样本)
                                       → label_01_60 (窗口3正样本)
                                                 → label_11 (延迟正样本)
        168h后仍未转化 ─────────────────────────────────────→ label_10 (真负样本)
        
        14 维标签格式:
        -------------
        索引  名称              含义                              用途
        ----  ----              ----                              ----
        [0]   label_11          延迟正样本 (label=1, diff > delay) CV loss 正样本
        [1]   label_10          真负样本 (label=0)                 CV loss 负样本
        [2]   label_01_15       窗口1内正样本 (diff ≤ win1)        win1 loss
        [3]   label_01_30       窗口2内正样本 (win1 < diff ≤ win2) win2 loss (增量)
        [4]   label_01_60       窗口3内正样本 (win2 < diff ≤ win3) win3 loss (增量)
        [5]   label_01_30_sum   窗口1+2累积 (diff ≤ win2)          win2 loss (累积)
        [6]   label_01_60_sum   窗口1+2+3累积 (diff ≤ win3)        win3 loss (累积)
        [7]   label_01_30_mask  窗口2 mask (恒为1)                 保留
        [8]   label_01_60_mask  窗口3 mask (恒为1)                 保留
        [9]   label_00          未观测负样本 (未使用)              保留
        [10]  label_01          所有窗口内正样本                   CV loss 正样本
        [11]  label_11_15       延迟正 (delay < diff ≤ 2*win3)     SPM loss
        [12]  label_11_30       延迟正 (2*win3 < diff ≤ 3*win3)    SPM loss
        [13]  label_11_60       延迟正 (diff > 3*win3)             SPM loss
        
        损失函数使用:
        -----------
        - CV loss: 正样本 = label_01 + label_11, 负样本 = label_10
        - Window loss: 使用 label_01_xx_sum 作为累积正样本
        - SPM loss: 使用 label_11_xx 细分延迟正样本，构建负样本 mask
        
        输入要求:
        --------
        - df 必须包含 "label" 列 (0/1 二分类)
        - df 必须包含 "diff_hours" 列 (点击到转化的小时数，负样本为大值或 null)
        
        输出:
        -----
        - 新增 "defer_label" 列: JSON 字符串格式 "[0.0,1.0,0.0,...]"
        - 使用 JSON 字符串而非 array<float>，避免 MetaSpore Spark 类型问题
        """
        MovasLogger.add_log(content="Building DEFER labels (pure Spark SQL, no UDF)")
        
        # =====================================================================
        # Step 1: 预处理 diff_hours 列
        # =====================================================================
        # diff_hours: 点击到转化的时间间隔 (小时)
        # - 正样本: diff_hours = 转化时间 - 点击时间
        # - 负样本: diff_hours 通常为 null 或很大的值 (表示在归因窗口内未转化)
        # 将 null 填充为 9999.0，确保负样本不会被误判为窗口内正样本
        if "diff_hours" in df.columns:
            df = df.withColumn("diff_hours", F.col("diff_hours").cast("float"))
            df = df.fillna({'diff_hours': 9999.0})
        
        # =====================================================================
        # Step 2: 定义时间窗口参数
        # =====================================================================
        # win1, win2, win3: 三个观测窗口 (小时)
        # - v1 配置: 15min, 30min, 60min (转换为小时: 0.25, 0.5, 1.0)
        # - v2 配置: 24h, 48h, 72h
        win1, win2, win3 = self.win1_hours, self.win2_hours, self.win3_hours
        
        # delay: 归因窗口 (attribution window)
        # - 含义: 超过这个时间仍未转化，才认为是"真负样本"
        # - 设定: 默认为 7 * win3 (例如 win3=72h 时，delay=504h=21天)
        # - 原因: 广告转化可能有很长的延迟，需要足够长的归因窗口
        # - 可通过 config.defer_delay_hours 覆盖
        delay = self.params.get("defer_delay_hours", win3 * 7)
        
        MovasLogger.add_log(
            content=f"Windows: WIN1={win1}h, WIN2={win2}h, WIN3={win3}h, DELAY={delay}h"
        )
        
        # =====================================================================
        # Step 3: 定义样本分类条件
        # =====================================================================
        label_col = F.col("label")      # 原始标签: 1=转化, 0=未转化
        diff_col = F.col("diff_hours")  # 转化延迟时间
        
        # 基础分类
        is_positive = label_col == 1.0  # 最终转化的样本
        is_negative = label_col == 0.0  # 最终未转化的样本 (归因窗口结束后仍未转化)
        
        # 延迟正样本: 转化了，但转化时间超过最大观测窗口 (win3)
        # 这些样本在训练时被观测为"未转化"，但实际上后来转化了
        is_delayed = is_positive & (diff_col > win3)
        
        # 窗口内正样本: 转化了，且转化时间在各窗口内
        # in_win1: 0 < diff_hours <= win1 (快速转化)
        # in_win2: win1 < diff_hours <= win2 (中等延迟)
        # in_win3: win2 < diff_hours <= win3 (较长延迟但仍在观测窗口内)
        in_win1 = is_positive & (diff_col <= win1)
        in_win2 = is_positive & (diff_col > win1) & (diff_col <= win2)
        in_win3 = is_positive & (diff_col > win2) & (diff_col <= win3)
        in_any_win = in_win1 | in_win2 | in_win3  # 任意窗口内的正样本
        
        # =====================================================================
        # Step 4: 构建 14 维标签
        # =====================================================================
        # [0] label_11: 延迟正样本 (转化了但超过 win3)
        df = df.withColumn("_lbl_0", F.when(is_delayed, 1.0).otherwise(0.0))
        
        # [1] label_10: 真负样本 (归因窗口结束后仍未转化)
        df = df.withColumn("_lbl_1", F.when(is_negative, 1.0).otherwise(0.0))
        
        # [2] label_01_15: 窗口1内正样本 (diff <= win1)
        df = df.withColumn("_lbl_2", F.when(in_win1, 1.0).otherwise(0.0))
        
        # [3] label_01_30: 窗口2内正样本 (win1 < diff <= win2)，增量
        df = df.withColumn("_lbl_3", F.when(in_win2, 1.0).otherwise(0.0))
        
        # [4] label_01_60: 窗口3内正样本 (win2 < diff <= win3)，增量
        df = df.withColumn("_lbl_4", F.when(in_win3, 1.0).otherwise(0.0))
        
        # [5] label_01_30_sum: 窗口1+2累积正样本 (diff <= win2)
        df = df.withColumn("_lbl_5", F.when(in_win1 | in_win2, 1.0).otherwise(0.0))
        
        # [6] label_01_60_sum: 窗口1+2+3累积正样本 (diff <= win3)
        df = df.withColumn("_lbl_6", F.when(in_any_win, 1.0).otherwise(0.0))
        
        # [7] label_01_30_mask: 窗口2 mask (恒为1，保留字段)
        df = df.withColumn("_lbl_7", F.lit(1.0))
        
        # [8] label_01_60_mask: 窗口3 mask (恒为1，保留字段)
        df = df.withColumn("_lbl_8", F.lit(1.0))
        
        # [9] label_00: 未观测负样本 (窗口内未转化但归因未结束，当前实现不使用)
        df = df.withColumn("_lbl_9", F.lit(0.0))
        
        # [10] label_01: 所有窗口内正样本的汇总
        df = df.withColumn("_lbl_10", F.when(in_any_win, 1.0).otherwise(0.0))
        
        # [11-13] 延迟正样本的细分 (用于 SPM loss 构建负样本 mask)
        # 根据延迟时间将延迟正样本分成 3 段:
        # - label_11_15: win3 < diff <= 2*win3 (轻度延迟)
        # - label_11_30: 2*win3 < diff <= 3*win3 (中度延迟)
        # - label_11_60: diff > 3*win3 (重度延迟)
        df = df.withColumn("_lbl_11", 
            F.when(is_delayed & (diff_col <= win3 * 2), 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_12", 
            F.when(is_delayed & (diff_col > win3 * 2) & (diff_col <= win3 * 3), 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_13", 
            F.when(is_delayed & (diff_col > win3 * 3), 1.0).otherwise(0.0))
        
        # =====================================================================
        # Step 5: 组合成 JSON 字符串
        # =====================================================================
        # 为什么用 JSON 字符串而不是 array<float>?
        # - MetaSpore 的 PyTorchEstimator 在处理 ArrayType 时会触发额外的 Spark stage
        # - 新 stage 会启动新的 Python worker，但这些 worker 没有注册 PS agent
        # - 导致 "no ps agent registered for thread" 错误
        # - 使用 JSON 字符串可以避免这个问题，在 estimator 中再解析
        df = df.withColumn(
            "defer_label",
            F.concat(
                F.lit("["),
                F.concat_ws(",", *[F.col(f"_lbl_{i}").cast("string") for i in range(14)]),
                F.lit("]")
            )
        )
        
        # =====================================================================
        # Step 6: 清理临时列
        # =====================================================================
        for i in range(14):
            df = df.drop(f"_lbl_{i}")
        
        return df
    
    def _log_label_stats(self, df: DataFrame) -> None:
        """
        打印标签统计信息 - 支持 JSON 字符串格式的 defer_label
        """
        # 由于 defer_label 现在是 JSON 字符串，需要解析后统计
        # 使用 from_json 解析 JSON 字符串
        from pyspark.sql.types import ArrayType, DoubleType
        
        # 解析 JSON 字符串为数组
        df_parsed = df.withColumn(
            "defer_label_array",
            F.from_json(F.col("defer_label"), ArrayType(DoubleType()))
        )
        
        # 统计各类标签数量
        stats = df_parsed.select(
            F.sum(F.when(F.col("defer_label_array")[0] == 1.0, 1).otherwise(0)).alias("delayed_positive"),
            F.sum(F.when(F.col("defer_label_array")[1] == 1.0, 1).otherwise(0)).alias("true_negative"),
            F.sum(F.when(F.col("defer_label_array")[10] == 1.0, 1).otherwise(0)).alias("window_positive"),
        ).collect()[0]
        
        MovasLogger.add_log(
            content=f"DEFER Label Stats: delayed_positive={stats['delayed_positive']}, "
                    f"true_negative={stats['true_negative']}, "
                    f"window_positive={stats['window_positive']}"
        )
    
    def _train_model(self, train_dataset, model_in_path_current, model_out_path_current, model_version_current):
        """
        训练模型 - 完全重写父类方法，使用 defer_label 和 delay_win_select_loss
        """
        import metaspore as ms
        
        MovasLogger.add_log(content=f"Starting DEFER training for version: {model_version_current}")
        
        # 构建 DEFER 标签
        train_dataset = self._build_defer_labels(train_dataset)
        
        # 关键修复：cache 并触发 UDF 执行，确保在 PS agent 启动之前完成
        # 这样 Spark 不会在 PS agent 启动后再创建新的 Python worker
        train_dataset = train_dataset.cache()
        row_count = train_dataset.count()
        MovasLogger.add_log(content=f"Dataset cached with {row_count} rows")
        
        # 打印标签统计
        self._log_label_stats(train_dataset)
        
        # 保留原始 label 列（用于 loss 函数中的单值标签），将 defer_label 重命名为 label
        # 这样 minibatch 中会同时包含：
        # - label: 14 维 DEFER 标签 (JSON 字符串)
        # - cv_label: 原始二分类标签 (是否最终转化)
        train_dataset = train_dataset.withColumnRenamed("label", "cv_label")
        train_dataset = train_dataset.withColumnRenamed("defer_label", "label")
        
        # 构建 estimator，使用 DEFER 损失函数
        if not self.model_module:
            self._build_model_module()
        
        # 使用 DEFER 自定义损失函数
        from metaspore.algos.delay_feedback.defer_loss import delay_win_select_loss
        
        estimator = ms.PyTorchEstimator(
            module=self.model_module,
            worker_count=self.worker_count,
            server_count=self.server_count,
            model_in_path=model_in_path_current,
            model_out_path=model_out_path_current,
            model_export_path=None,
            model_version=model_version_current,
            experiment_name=self.experiment_name,
            input_label_column_name='label',  # 使用重命名后的 defer_label (JSON 字符串)
            loss_function=delay_win_select_loss,  # DEFER 自定义损失函数
            metric_update_interval=1000
        )
        estimator.updater = ms.AdamTensorUpdater(self.adam_learning_rate)
        
        MovasLogger.add_log(content=f"DEFER training: Model In: {model_in_path_current}, Model Out: {model_out_path_current}")
        
        model = estimator.fit(train_dataset)
        self.trained_model_path = model_out_path_current
        return model
    
    def _run_evaluation_phase(self):
        """
        重写评估阶段，使用 DEFER 标签
        """
        MovasLogger.add_log(content="Starting DEFER evaluation phase")
        
        # 读取验证数据
        validation_date = self.params.get("validation_date")
        if validation_date is None:
            MovasLogger.add_log(content="No validation_date specified, skipping evaluation")
            return
        
        date_str = validation_date.strftime("%Y-%m-%d") if hasattr(validation_date, 'strftime') else str(validation_date)
        
        # 读取数据
        df = self._read_dataset_by_date(self.train_path_prefix, date_str)
        
        # 构建 DEFER 标签
        df = self._build_defer_labels(df)
        
        # 过滤业务类型
        if "business_type" in df.columns:
            df = df.filter(F.col("business_type").isin(FILTER_BUSINESS_TYPES))
        
        # 调用父类评估
        super()._run_evaluation_phase()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DEFER Training')
    parser.add_argument('--conf', type=str, required=True, help='config file path')
    parser.add_argument('--name', type=str, default='defer', help='experiment name')
    parser.add_argument('--eval_keys', type=str, default='business_type', help='evaluation keys')
    parser.add_argument('--validation', type=bool, default=False, help='validation only mode')
    parser.add_argument('--model_date', type=str, help='model date for validation')
    parser.add_argument('--sample_date', type=str, help='sample date for validation')
    args = parser.parse_args()
    
    print(f"DEFER Training | Config: {args.conf}")
    
    trainer = DeferTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args)
    MovasLogger.save_to_local()
