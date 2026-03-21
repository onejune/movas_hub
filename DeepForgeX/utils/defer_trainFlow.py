#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEFER TrainFlow - 延迟反馈建模训练流程

基于 dnn_trainFlow.py 风格重构，主要改动:
1. 损失函数统一走 get_loss_function (从 config.yaml 配置)
2. main 函数使用 parse_args 静态方法
3. 代码风格与 dnn_trainFlow.py 保持一致
"""
import os
import sys
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

# MetaSpore - 使用 DeepForgeX 目录
sys.path.insert(0, "/mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/MetaSpore/python")

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

# 过滤业务类型
FILTER_BUSINESS_TYPES = ["aecps", "shein", "lazada_rta", "lazada_cps", "aedsp", "aerta"]


# ============================================================================
# DeferTrainFlow
# ============================================================================

class DeferTrainFlow(MsModelTrainFlow):
    """
    DEFER (Delayed Feedback) 训练流程
    
    继承 MsModelTrainFlow，重写关键方法:
    - _read_dataset_by_date: 自动检测分区格式，保留 diff_hours
    - _build_model_module: 构建 DEFER 模型 (WinAdaptDNN/DeferDNN)
    - _train_model: 添加 DEFER 标签构建，使用配置的损失函数
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
        
        MovasLogger.log(f"[DEFER] Windows: {self.win1_hours}h, {self.win2_hours}h, {self.win3_hours}h, Delay: {self.delay_hours}h")
        MovasLogger.log(f"[DEFER] Model type: {self.defer_model_type}, Loss: {self.loss_func}")
    
    # ========================================================================
    # 数据读取
    # ========================================================================
    
    def _read_dataset_by_date(self, base_path: str, date_str: str) -> DataFrame:
        """
        读取数据 - 自动检测分区格式 (part= 或 sample_date=)
        保留 diff_hours 列用于 DEFER 标签构建
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
        
        df = self.random_sample(df)
        df = df.fillna('')
        
        MovasLogger.log(f"[DEFER] Loaded {len(df.columns)} columns")
        return df
    
    # ========================================================================
    # 模型构建
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
    # DEFER 标签构建
    # ========================================================================
    
    def _build_defer_labels(self, df: DataFrame) -> DataFrame:
        """
        构建 DEFER 14 维标签 - 纯 Spark SQL，避免 Python UDF
        
        14 维标签格式:
        [0]  label_11: 延迟正样本 (label=1, diff > delay)
        [1]  label_10: 真负样本 (label=0)
        [2]  label_01_15: 窗口1内正样本
        [3]  label_01_30: 窗口2内正样本 (增量)
        [4]  label_01_60: 窗口3内正样本 (增量)
        [5]  label_01_30_sum: 窗口1+2累积
        [6]  label_01_60_sum: 窗口1+2+3累积
        [7]  label_01_30_mask: 恒为1
        [8]  label_01_60_mask: 恒为1
        [9]  label_00: 未使用
        [10] label_01: 所有窗口内正样本
        [11-13] label_11_xx: 延迟正样本细分
        
        输出: 新增 defer_label 列 (JSON 字符串)
        """
        MovasLogger.log("[DEFER] Building 14-dim labels (Spark SQL)")
        
        # 预处理 diff_hours
        if "diff_hours" in df.columns:
            df = df.withColumn("diff_hours", F.col("diff_hours").cast("float"))
            df = df.fillna({'diff_hours': 9999.0})
        
        win1, win2, win3 = self.win1_hours, self.win2_hours, self.win3_hours
        delay = self.delay_hours
        
        MovasLogger.log(f"[DEFER] WIN1={win1}h, WIN2={win2}h, WIN3={win3}h, DELAY={delay}h")
        
        # 条件定义
        label_col = F.col("label")
        diff_col = F.col("diff_hours")
        
        is_positive = label_col == 1.0
        is_negative = label_col == 0.0
        is_delayed = is_positive & (diff_col > delay)
        
        in_win1 = is_positive & (diff_col <= win1)
        in_win2 = is_positive & (diff_col > win1) & (diff_col <= win2)
        in_win3 = is_positive & (diff_col > win2) & (diff_col <= win3)
        in_any_win = in_win1 | in_win2 | in_win3
        
        # 构建 14 维标签
        df = df.withColumn("_lbl_0", F.when(is_delayed, 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_1", F.when(is_negative, 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_2", F.when(in_win1, 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_3", F.when(in_win2, 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_4", F.when(in_win3, 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_5", F.when(in_win1 | in_win2, 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_6", F.when(in_any_win, 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_7", F.lit(1.0))
        df = df.withColumn("_lbl_8", F.lit(1.0))
        df = df.withColumn("_lbl_9", F.lit(0.0))
        df = df.withColumn("_lbl_10", F.when(in_any_win, 1.0).otherwise(0.0))
        
        # 延迟正样本细分
        df = df.withColumn("_lbl_11", 
            F.when(is_delayed & (diff_col <= delay + win3), 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_12", 
            F.when(is_delayed & (diff_col > delay + win3) & (diff_col <= delay + win3 * 2), 1.0).otherwise(0.0))
        df = df.withColumn("_lbl_13", 
            F.when(is_delayed & (diff_col > delay + win3 * 2), 1.0).otherwise(0.0))
        
        # 组合成 JSON 字符串 (避免 ArrayType 触发 PS agent 问题)
        df = df.withColumn(
            "defer_label",
            F.concat(
                F.lit("["),
                F.concat_ws(",", *[F.col(f"_lbl_{i}").cast("string") for i in range(14)]),
                F.lit("]")
            )
        )
        
        # 清理临时列
        for i in range(14):
            df = df.drop(f"_lbl_{i}")
        
        return df
    
    def _log_label_stats(self, df: DataFrame) -> None:
        """打印标签统计"""
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
            f"[DEFER] Labels: delayed_pos={stats['delayed_pos']}, "
            f"true_neg={stats['true_neg']}, window_pos={stats['window_pos']}"
        )
    
    # ========================================================================
    # 训练
    # ========================================================================
    
    def _train_model(self, train_dataset, model_in_path_current, model_out_path_current, model_version_current):
        """
        训练模型 - 使用 DEFER 标签和配置的损失函数
        """
        if not self.model_module:
            self._build_model_module()
        
        MovasLogger.log(f"[DEFER] Training version: {model_version_current}")
        
        # 构建 DEFER 标签
        train_dataset = self._build_defer_labels(train_dataset)
        
        # Cache 并触发执行，避免 PS agent 问题
        train_dataset = train_dataset.cache()
        row_count = train_dataset.count()
        MovasLogger.log(f"[DEFER] Dataset cached: {row_count:,} rows")
        
        # 打印标签统计
        self._log_label_stats(train_dataset)
        
        # 标签列处理: label -> cv_label, defer_label -> label
        train_dataset = train_dataset.withColumnRenamed("label", "cv_label")
        train_dataset = train_dataset.withColumnRenamed("defer_label", "label")
        
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
    
    # ========================================================================
    # 静态方法
    # ========================================================================
    
    @staticmethod
    def parse_args():
        """解析命令行参数 - 与 dnn_trainFlow.py 保持一致"""
        parser = argparse.ArgumentParser(description='DEFER Training')
        parser.add_argument('--conf', type=str, required=True, help='config file path')
        parser.add_argument('--name', type=str, required=True, help='experiment name')
        parser.add_argument('--eval_keys', type=str, required=True, help='evaluation keys')
        parser.add_argument('--validation', type=bool, default=False, help='validation only mode')
        parser.add_argument('--model_date', type=str, help='model date for validation')
        parser.add_argument('--sample_date', type=str, help='sample date for validation')
        return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    args = DeferTrainFlow.parse_args()
    print(f'DeferTrainFlow: args={args}')
    
    trainer = DeferTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args)
    MovasLogger.save_to_local()
