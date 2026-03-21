#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEFER TrainFlow - 延迟反馈建模训练流程

基于 dnn_trainFlow.py 风格重构:
1. 损失函数统一走 get_loss_function (从 config.yaml 配置)
2. main 函数使用 parse_args 静态方法
3. 代码风格与 dnn_trainFlow.py 保持一致
4. _read_dataset_by_date 处理 sample 和 label，train/validation 共用
5. label (1维) 用于 AUC 评估，defer_label (14维JSON) 用于 loss
"""
import os
import sys

# MetaSpore - 使用 DeepForgeX 目录
sys.path.insert(0, "/mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/MetaSpore/python")

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Spark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ArrayType, StringType, DoubleType

# PyTorch
import torch
import torch.nn as nn

import metaspore as ms
from metaspore.algos.delay_feedback.defer_models import create_defer_model
from metaspore.loss_utils import get_loss_function

# 基类
from dnn_trainFlow import MsModelTrainFlow
from movas_logger import MovasLogger


# ============================================================================
# 常量
# ============================================================================

# 默认时间窗口 (小时)
DEFAULT_WIN1_HOURS = 24.0
DEFAULT_WIN2_HOURS = 48.0
DEFAULT_WIN3_HOURS = 72.0
DEFAULT_DELAY_HOURS = 168.0  # 归因窗口 7 天


# ============================================================================
# DeferTrainFlow
# ============================================================================

MovasLogger.set_debug_mode(True)

class DeferTrainFlow(MsModelTrainFlow):
    """
    DEFER (Delayed Feedback) 训练流程
    
    继承 MsModelTrainFlow，重写关键方法:
    - _read_dataset_by_date: 自动检测分区格式，构建 DEFER 标签
    - _build_model_module: 构建 DEFER 模型 (WinAdaptDNN/DeferDNN)
    - _train_model: 使用配置的损失函数训练
    
    标签约定:
    - label: 原始 1 维标签，用于 validation 的 AUC/PCOC 评估
    - defer_label: 14 维 JSON 字符串，用于 DEFER loss 计算
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self._init_defer_params()
    
    def _init_defer_params(self):
        """初始化 DEFER 相关参数"""
        # 时间窗口配置
        defer_windows = self.params.get("defer_windows", 
            [DEFAULT_WIN1_HOURS, DEFAULT_WIN2_HOURS, DEFAULT_WIN3_HOURS])
        self.win1_hours = float(defer_windows[0]) if len(defer_windows) > 0 else DEFAULT_WIN1_HOURS
        self.win2_hours = float(defer_windows[1]) if len(defer_windows) > 1 else DEFAULT_WIN2_HOURS
        self.win3_hours = float(defer_windows[2]) if len(defer_windows) > 2 else DEFAULT_WIN3_HOURS
        self.delay_hours = float(self.params.get("defer_delay_hours", DEFAULT_DELAY_HOURS))
        
        # 模型类型
        self.defer_model_type = self.params.get("model_type", "WinAdaptDNN")
        
        # 标签版本: v1 (14维) 或 v2 (8维)
        # defer_loss -> v1, defer_loss_v2 -> v2
        self.label_version = "v2" if self.loss_func == "defer_loss_v2" else "v1"
        
        MovasLogger.log(f"[DEFER] Windows: {self.win1_hours}h, {self.win2_hours}h, {self.win3_hours}h, Delay: {self.delay_hours}h")
        MovasLogger.log(f"[DEFER] Model type: {self.defer_model_type}, Loss: {self.loss_func}, Label version: {self.label_version}")
    
    # ========================================================================
    # 数据读取 (重写基类方法)
    # ========================================================================
    
    def _read_dataset_by_date(self, base_path: str, date_str: str) -> DataFrame:
        """
        读取数据并构建 DEFER 标签
        
        处理流程:
        1. 自动检测分区格式 (part= 或 sample_date=)
        2. 保留 diff_hours 列用于标签构建
        3. 调用 random_sample 采样
        4. 构建 defer_label (14维 JSON)
        5. 保留 label (1维) 用于评估
        
        Args:
            base_path: 数据基础路径
            date_str: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            DataFrame: 包含 label 和 defer_label 的数据集
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
        
        MovasLogger.log(f"[DEFER] Reading Parquet: {data_path}")
        
        df = self.spark_session.read.parquet(data_path)
        
        # 选择需要的列 (包含 diff_hours)
        used_cols = list(self.used_fea_list) + ['diff_hours']
        used_cols = [c for c in used_cols if c in df.columns]
        df = df.select(*used_cols)
        
        # 类型转换
        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            elif col_name == 'diff_hours':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        
        # 调用基类的采样方法
        df = self.random_sample(df)
        df = df.fillna('')
        
        # 根据 loss 函数选择标签版本
        if self.label_version == "v2":
            df = self._build_defer_labels_v2(df)  # 8 维标签
        else:
            df = self._build_defer_labels(df)     # 14 维标签
        
        MovasLogger.log(f"[DEFER] Loaded {len(df.columns)} columns")
        return df
    
    # ========================================================================
    # DEFER 标签构建
    # ========================================================================
    
    def _build_defer_labels(self, df: DataFrame) -> DataFrame:
        """
        构建 DEFER 14 维标签 - 纯 Spark SQL，避免 Python UDF
        
        ============================================================================
        DEFER (Delayed Feedback) 模型标签设计说明
        ============================================================================
        
        背景:
        - 广告转化存在延迟反馈问题: 用户点击广告后，转化可能在几小时甚至几天后发生
        - 传统做法是等待足够长时间 (如7天) 才能确定标签，但这会导致数据延迟
        - DEFER 方法通过时间窗口对样本分类，结合概率建模来处理延迟反馈
        
        样本分类 (基于 label 和 diff_hours):
        - label: 最终转化标签 (0=未转化, 1=转化)
        - diff_hours: 从点击到转化的时间差 (小时)
        
        +-----------------+------------------+-----------------------------------+
        | 样本类型        | 条件              | 说明                              |
        +-----------------+------------------+-----------------------------------+
        | label_01        | label=1,         | 窗口内正样本: 在观测窗口内转化    |
        | (window pos)    | diff <= win3     | 这些样本可以确定是正样本          |
        +-----------------+------------------+-----------------------------------+
        | label_11        | label=1,         | 延迟正样本: 超过观测窗口才转化    |
        | (delayed pos)   | diff > delay     | 训练时这些样本被误判为负样本      |
        +-----------------+------------------+-----------------------------------+
        | label_10        | label=0          | 真负样本: 最终未转化              |
        | (true neg)      |                  | 需要足够长的归因窗口才能确定      |
        +-----------------+------------------+-----------------------------------+
        | label_00        | label=1,         | 未观测完样本: 在窗口外但未超过    |
        | (unobserved)    | win3<diff<=delay | 归因窗口，无法确定最终标签        |
        +-----------------+------------------+-----------------------------------+
        
        时间窗口示意 (以 24/48/72h 窗口, 168h 归因为例):
        
        点击时刻 -----> 时间流逝 ----->
        |----win1(24h)----|----win2(48h)----|----win3(72h)----|----delay(168h)----|
        |<-- label_01_w1 ->|<-- label_01_w2 ->|<-- label_01_w3 ->|                  |
        |<-------------- label_01 (窗口内正) ---------------->|                  |
                                                              |<-- label_00 -->|
                                                                               |<-- label_11 (延迟正) -->
        
        14 维标签格式:
        ============================================================================
        索引  名称              计算公式                          用途
        ============================================================================
        [0]   label_11          label=1 AND diff>delay            延迟正样本 (SPM loss)
        [1]   label_10          label=0                           真负样本 (CV/SPM loss)
        [2]   label_01_win1     label=1 AND diff<=win1            窗口1正样本 (time loss)
        [3]   label_01_win2     label=1 AND win1<diff<=win2       窗口2正样本 (增量)
        [4]   label_01_win3     label=1 AND win2<diff<=win3       窗口3正样本 (增量)
        [5]   label_01_sum12    label_01_win1 OR label_01_win2    窗口1+2累积
        [6]   label_01_sum123   label_01_win1 OR _win2 OR _win3   窗口1+2+3累积
        [7]   mask_win2         1.0 (恒定)                        窗口2 mask
        [8]   mask_win3         1.0 (恒定)                        窗口3 mask
        [9]   reserved          0.0 (保留)                        未使用
        [10]  label_01          同 [6]                            所有窗口内正样本
        [11]  label_11_w1       延迟正 AND diff<=delay+win3       延迟正细分1
        [12]  label_11_w2       延迟正 AND delay+win3<diff<=...   延迟正细分2
        [13]  label_11_w3       延迟正 AND diff>delay+2*win3      延迟正细分3
        ============================================================================
        
        输出:
        - defer_label: 新增列 (JSON 字符串 "[0.0,1.0,...]")，用于 loss 计算
        - label: 保持原始 1 维标签 (0/1)，用于 AUC 评估
        
        注意:
        - 使用 JSON 字符串而非 ArrayType，避免 Spark ArrayType 触发 PS agent 问题
        - loss 函数从 minibatch['defer_label'] 解析 JSON 获取 14 维标签
        """
        MovasLogger.log("[DEFER] Building 14-dim labels (Spark SQL)")
        
        # ========== Step 1: 预处理 diff_hours ==========
        # diff_hours 表示从点击到转化的时间差 (小时)
        # 如果为 null (未转化)，填充为 9999.0 (远大于任何窗口)
        if "diff_hours" in df.columns:
            df = df.withColumn("diff_hours", F.col("diff_hours").cast("float"))
            df = df.fillna({'diff_hours': 9999.0})
        
        # 时间窗口参数 (从 config 读取)
        win1, win2, win3 = self.win1_hours, self.win2_hours, self.win3_hours
        delay = self.delay_hours
        
        MovasLogger.log(f"[DEFER] WIN1={win1}h, WIN2={win2}h, WIN3={win3}h, DELAY={delay}h")
        
        # ========== Step 2: 定义分类条件 ==========
        label_col = F.col("label")
        diff_col = F.col("diff_hours")
        
        # 基础条件
        is_positive = label_col == 1.0  # 最终转化
        is_negative = label_col == 0.0  # 最终未转化
        
        # 延迟正样本: 转化了，但超过归因窗口才转化
        is_delayed = is_positive & (diff_col > delay)
        
        # 窗口内正样本: 转化了，且在对应窗口内转化
        in_win1 = is_positive & (diff_col <= win1)                        # 0 ~ win1
        in_win2 = is_positive & (diff_col > win1) & (diff_col <= win2)    # win1 ~ win2
        in_win3 = is_positive & (diff_col > win2) & (diff_col <= win3)    # win2 ~ win3
        in_any_win = in_win1 | in_win2 | in_win3                          # 0 ~ win3
        
        # ========== Step 3: 构建 14 维标签 ==========
        # [0] label_11: 延迟正样本
        df = df.withColumn("_lbl_0", F.when(is_delayed, 1.0).otherwise(0.0))
        
        # [1] label_10: 真负样本
        df = df.withColumn("_lbl_1", F.when(is_negative, 1.0).otherwise(0.0))
        
        # [2-4] 各窗口正样本 (增量，不重叠)
        df = df.withColumn("_lbl_2", F.when(in_win1, 1.0).otherwise(0.0))  # win1 内
        df = df.withColumn("_lbl_3", F.when(in_win2, 1.0).otherwise(0.0))  # win1~win2
        df = df.withColumn("_lbl_4", F.when(in_win3, 1.0).otherwise(0.0))  # win2~win3
        
        # [5-6] 累积窗口正样本
        df = df.withColumn("_lbl_5", F.when(in_win1 | in_win2, 1.0).otherwise(0.0))  # win1+win2
        df = df.withColumn("_lbl_6", F.when(in_any_win, 1.0).otherwise(0.0))         # win1+win2+win3
        
        # [7-8] mask (恒为 1.0，用于 loss 计算)
        df = df.withColumn("_lbl_7", F.lit(1.0))  # mask_win2
        df = df.withColumn("_lbl_8", F.lit(1.0))  # mask_win3
        
        # [9] reserved (保留位，未使用)
        df = df.withColumn("_lbl_9", F.lit(0.0))
        
        # [10] label_01: 所有窗口内正样本 (同 [6])
        df = df.withColumn("_lbl_10", F.when(in_any_win, 1.0).otherwise(0.0))
        
        # [11-13] 延迟正样本细分 (用于更精细的 loss 计算)
        # 细分为: delay~delay+win3, delay+win3~delay+2*win3, >delay+2*win3
        df = df.withColumn("_lbl_11", 
            F.when(is_delayed & (diff_col <= delay + win3), 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_12", 
            F.when(is_delayed & (diff_col > delay + win3) & (diff_col <= delay + win3 * 2), 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_13", 
            F.when(is_delayed & (diff_col > delay + win3 * 2), 1.0).otherwise(0.0))
        
        # ========== Step 4: 组合成 JSON 字符串 ==========
        # 使用 JSON 字符串而非 ArrayType，避免 Spark ArrayType 触发 PS agent 注册问题
        # 格式: "[0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0]"
        df = df.withColumn(
            "defer_label",
            F.concat(
                F.lit("["),
                F.concat_ws(",", *[F.col(f"_lbl_{i}").cast("string") for i in range(14)]),
                F.lit("]")
            )
        )
        
        # ========== Step 5: 清理临时列 ==========
        for i in range(14):
            df = df.drop(f"_lbl_{i}")
        
        # 打印标签统计
        self._log_label_stats(df)
        
        return df
    
    def _build_defer_labels_v2(self, df: DataFrame) -> DataFrame:
        """
        构建 DEFER 8 维标签 (v2 版本) - 用于 delay_win_select_loss_v2
        
        ============================================================================
        与 v1 (14 维) 的区别
        ============================================================================
        v1 (14维): 复杂的 SPM loss，需要延迟正样本细分、累积窗口标签等
        v2 (8维):  简化版，只需要各窗口标签 + observable mask
        
        ============================================================================
        8 维标签格式
        ============================================================================
        索引  名称              计算公式                          用途
        ----  ----              ----                              ----
        [0]   label_win1        label=1 AND diff<=win1            窗口1标签 (累积)
        [1]   label_win2        label=1 AND diff<=win2            窗口2标签 (累积)
        [2]   label_win3        label=1 AND diff<=win3            窗口3标签 (累积)
        [3]   label_oracle      label=1 AND diff<=delay           最终转化标签
        [4]   observable_win1   sample_age >= win1                窗口1可观察 mask
        [5]   observable_win2   sample_age >= win2                窗口2可观察 mask
        [6]   observable_win3   sample_age >= win3                窗口3可观察 mask
        [7]   observable_oracle 1.0 (恒定)                        oracle 可观察 mask
        
        ============================================================================
        observable mask 说明
        ============================================================================
        样本年龄 (sample_age) = 当前时间 - 样本时间
        
        如果样本年龄 < 窗口时间，则该样本在该窗口上不可观察:
        - 例如: 样本年龄=12h，win1=24h，则 observable_win1=0
        - 这意味着我们不知道该样本是否会在 24h 内转化
        
        在 loss 计算中，不可观察的样本会被 mask 掉，不参与梯度计算
        
        注意: 当前实现假设所有样本都已经过了足够长的时间 (observable=1)
        如果需要处理实时数据，需要传入 sample_age 列
        """
        MovasLogger.log("[DEFER] Building 8-dim labels v2 (Spark SQL)")
        
        # ========== Step 1: 预处理 diff_hours ==========
        if "diff_hours" in df.columns:
            df = df.withColumn("diff_hours", F.col("diff_hours").cast("float"))
            df = df.fillna({'diff_hours': 9999.0})
        
        win1, win2, win3 = self.win1_hours, self.win2_hours, self.win3_hours
        delay = self.delay_hours
        
        MovasLogger.log(f"[DEFER v2] WIN1={win1}h, WIN2={win2}h, WIN3={win3}h, DELAY={delay}h")
        
        # ========== Step 2: 定义分类条件 ==========
        label_col = F.col("label")
        diff_col = F.col("diff_hours")
        
        is_positive = label_col == 1.0
        
        # 累积窗口标签 (注意: v2 是累积的，不是增量的)
        in_win1 = is_positive & (diff_col <= win1)    # 0 ~ win1
        in_win2 = is_positive & (diff_col <= win2)    # 0 ~ win2
        in_win3 = is_positive & (diff_col <= win3)    # 0 ~ win3
        in_delay = is_positive & (diff_col <= delay)  # 0 ~ delay (oracle)
        
        # ========== Step 3: 构建 8 维标签 ==========
        # [0-3] 各窗口标签 (累积)
        df = df.withColumn("_lbl_0", F.when(in_win1, 1.0).otherwise(0.0))   # label_win1
        df = df.withColumn("_lbl_1", F.when(in_win2, 1.0).otherwise(0.0))   # label_win2
        df = df.withColumn("_lbl_2", F.when(in_win3, 1.0).otherwise(0.0))   # label_win3
        df = df.withColumn("_lbl_3", F.when(in_delay, 1.0).otherwise(0.0))  # label_oracle
        
        # [4-7] observable mask
        # 当前实现: 假设所有样本都已经过了足够长的时间
        # TODO: 如果需要处理实时数据，可以从 sample_age 列计算
        df = df.withColumn("_lbl_4", F.lit(1.0))  # observable_win1
        df = df.withColumn("_lbl_5", F.lit(1.0))  # observable_win2
        df = df.withColumn("_lbl_6", F.lit(1.0))  # observable_win3
        df = df.withColumn("_lbl_7", F.lit(1.0))  # observable_oracle
        
        # ========== Step 4: 组合成 JSON 字符串 ==========
        df = df.withColumn(
            "defer_label",
            F.concat(
                F.lit("["),
                F.concat_ws(",", *[F.col(f"_lbl_{i}").cast("string") for i in range(8)]),
                F.lit("]")
            )
        )
        
        # ========== Step 5: 清理临时列 ==========
        for i in range(8):
            df = df.drop(f"_lbl_{i}")
        
        # 打印标签统计
        self._log_label_stats_v2(df)
        
        return df
    
    def _log_label_stats(self, df: DataFrame) -> None:
        """打印 14 维标签统计 (v1)"""
        df_parsed = df.withColumn(
            "defer_label_array",
            F.from_json(F.col("defer_label"), ArrayType(DoubleType()))
        )
        
        stats = df_parsed.select(
            F.sum(F.when(F.col("defer_label_array")[0] == 1.0, 1).otherwise(0)).alias("delayed_pos"),
            F.sum(F.when(F.col("defer_label_array")[1] == 1.0, 1).otherwise(0)).alias("true_neg"),
            F.sum(F.when(F.col("defer_label_array")[10] == 1.0, 1).otherwise(0)).alias("window_pos"),
        ).collect()[0]
        
        MovasLogger.log(
            f"[DEFER v1] Labels: delayed_pos={stats['delayed_pos']}, "
            f"true_neg={stats['true_neg']}, window_pos={stats['window_pos']}"
        )
    
    def _log_label_stats_v2(self, df: DataFrame) -> None:
        """打印 8 维标签统计 (v2)"""
        df_parsed = df.withColumn(
            "defer_label_array",
            F.from_json(F.col("defer_label"), ArrayType(DoubleType()))
        )
        
        stats = df_parsed.select(
            F.sum(F.when(F.col("defer_label_array")[0] == 1.0, 1).otherwise(0)).alias("win1_pos"),
            F.sum(F.when(F.col("defer_label_array")[1] == 1.0, 1).otherwise(0)).alias("win2_pos"),
            F.sum(F.when(F.col("defer_label_array")[2] == 1.0, 1).otherwise(0)).alias("win3_pos"),
            F.sum(F.when(F.col("defer_label_array")[3] == 1.0, 1).otherwise(0)).alias("oracle_pos"),
        ).collect()[0]
        
        MovasLogger.log(
            f"[DEFER v2] Labels: win1_pos={stats['win1_pos']}, win2_pos={stats['win2_pos']}, "
            f"win3_pos={stats['win3_pos']}, oracle_pos={stats['oracle_pos']}"
        )
    
    # ========================================================================
    # 模型构建 (重写基类方法)
    # ========================================================================
    
    def _build_model_module(self):
        """
        构建 DEFER 模型
        
        支持模型类型 (通过 config.model_type 配置):
        - WinAdaptDNN: 单输出层 4 头，适合长窗口 (24/48/72h)
        - DeferDNN: 多输出头，适合短窗口 (15/30/60min)
        """
        MovasLogger.log(f"[DEFER] Building model: {self.defer_model_type}")
        
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
        
        self.model_module = create_defer_model(self.defer_model_type, **model_kwargs)
        self.configed_model = self.defer_model_type
    
    # ========================================================================
    # 训练 (重写基类方法)
    # ========================================================================
    
    def _train_model(self, train_dataset, model_in_path_current, model_out_path_current, model_version_current):
        """
        训练模型 - 使用 DEFER 标签和配置的损失函数
        
        标签处理:
        - 训练时: label -> cv_label (保存原始标签), defer_label -> label (用于 loss)
        - 评估时: 基类 _evaluate_model 使用 label 列，我们在 validation 时不做标签切换
        """
        if not self.model_module:
            self._build_model_module()
        
        MovasLogger.log(f"[DEFER] Training version: {model_version_current}")
        
        # Cache 并触发执行，避免 PS agent 问题
        train_dataset = train_dataset.cache()
        row_count = train_dataset.count()
        MovasLogger.log(f"[DEFER] Dataset cached: {row_count:,} rows")
        
        # 不需要切换标签了！
        # loss 函数直接从 minibatch['defer_label'] 获取 14 维标签
        # label 列 (1 维) 保留用于 AUC 评估
        
        # 获取损失函数 (统一走 get_loss_function)
        loss_func = get_loss_function(self.loss_func)
        if not loss_func:
            raise ValueError(f"Invalid loss function: {self.loss_func}")
        
        self.agent_class = ms.PyTorchAgent
        
        estimator = ms.PyTorchEstimator(
            module=self.model_module,
            agent_class=self.agent_class,
            worker_count=self.worker_count,
            server_count=self.server_count,
            model_in_path=model_in_path_current,
            model_out_path=model_out_path_current,
            model_export_path=None,
            model_version=model_version_current,
            experiment_name=self.experiment_name,
            input_label_column_name='label',
            loss_function=loss_func,
            metric_update_interval=1000,
        )
        estimator.updater = ms.AdamTensorUpdater(self.adam_learning_rate)
        
        MovasLogger.log(
            f"[DEFER] Training: model_in={model_in_path_current}, "
            f"model_out={model_out_path_current}, loss={self.loss_func}"
        )
        
        model = estimator.fit(train_dataset)
        self.trained_model_path = model_out_path_current
        return model
    
# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    args = DeferTrainFlow.parse_args()
    print(f'DeferTrainFlow: args={args}')
    
    trainer = DeferTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args)
    MovasLogger.save_to_local()
