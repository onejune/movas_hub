# Delay-Feedback learning framework

import os, sys, random, traceback, json
import numpy as np
import shutil, inspect, math
from operator import itemgetter
from pyspark.sql.types import StringType,ArrayType, IntegerType, StructField, StructType, MapType
from pyspark.sql import functions as F
from pyspark.sql.functions import create_map, coalesce, lit
from metrics_eval import compute_auc_and_pcoc_regression
from pyspark.sql.functions import rand, to_json, col
from datetime import datetime, timedelta
import time
from tabulate import tabulate
from collections import namedtuple
from typing import List, Any, Optional
from dnn_trainFlow import MsModelTrainFlow
import metaspore as ms
from metaspore.algos.star_net import STARModel

from metaspore.algos.delay_feedback import DELFWideDeep
from metaspore.loss_utils import get_loss_function
from metaspore.loss_utils import LossUtils

from movas_logger import MovasLogger, how_much_time

MovasLogger.set_debug_mode(False)

class DELFModelTrainFlow(MsModelTrainFlow):
    def __init__(self, config_path):
        super().__init__(config_path)
    
    def _build_model_module(self):
        configed_model = self.params.get('model_type')
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        if configed_model == "delf": 
            self.model_module = DELFWideDeep(
                use_wide=self.use_wide,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                wide_embedding_dim=self.embedding_size,
                deep_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                deep_combine_schema_path=self.combine_schema_path,
                dnn_hidden_units=self.dnn_hidden_units,
                dnn_hidden_activations="relu",
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        else:
            raise ValueError("Invalid model type specified.")
        self.configed_model = configed_model

    def random_sample(self, df):
        df_filtered = df.filter(
            (F.col("label") == 1) |  # 正样本全部保留
            (
                (F.col("label") == 0) &  # 负样本进行条件采样
                F.when(
                    (F.col("business_type") == 'shein') | (F.col("objective_type") == 'SALES_WEBSITE'),
                    F.rand(seed=42) < 0.01  # shein负样本采样1%
                )
                .otherwise(
                    F.rand(seed=42) < 0.1   # 其他负样本采样10%
                )
            )
        )
        return df_filtered

    def _preprocess(self):
        self.type_to_id_map = {
            'shein': 0,
            'shopee_cps': 1,
            'aecps': 2,
            'aerta': 3,
            'everglowly': 4,
            'lazada_rta': 3,
            'lazada_cps': 1,
            'ttshop': 4,
            'miravia_rta': 3,
            'liongame': 4,
            'cyberclickshecurve': 4,
            'detroitrain': 4,
            'swiftlink': 4,
            'saker': 4,
            'starpony': 4,
            'alibaba': 2,
            'mena_ae_cps': 1
        }
        domains_cnt = len(list(set(self.type_to_id_map.values())))
        MovasLogger.log(f'type_to_id_map = {self.type_to_id_map}, domains_cnt = {domains_cnt}')
        self.domains_cnt = domains_cnt + 1
        
        self.task_to_id_map = {
            'purchase': 0,
            'atc': 1,
            'content_view': 2,
            'open': 3
        }

        task_cnt = len(list(set(self.task_to_id_map.values())))
        self.tasks_cnt = task_cnt
        MovasLogger.log(f'task_to_id_map = {self.task_to_id_map}, task_cnt = {task_cnt}, task_weights_dict = {self.task_weights_dict}')

    def _read_dataset_by_date(self, base_path, date_str):
        data_path = os.path.join(base_path, f"part={date_str}")
        df = self.spark_session.read.parquet(data_path)
        MovasLogger.log(f"before_sample_cnt: {df.count()}")

        used_fea_list = self.used_fea_list + ['mul_labels', 'diff_hours', 'attributeWindow']
        df = df.select(*used_fea_list)

        df = self.random_sample(df)
        MovasLogger.log(f"final_sample_cnt: {df.count()}")

        df = df.withColumn("mul_labels", to_json(col("mul_labels")))
        df = df.withColumn('diff_hours', F.col('diff_hours').cast('float'))
        df = df.withColumn('label', F.col('label').cast('int'))
        df = df.withColumn('attributeWindow', F.col('attributeWindow').cast('float'))

        # 构造 observed_conversion：只有 label=1 且 diff_hours <= 72h 才算观测到转化
        # observed_conversion会做为Convert Propensity 的学习 label
        df = df.withColumn(
            "observed_conversion",
            F.when((col("label") == 1) & (col("diff_hours") <= 72), 1).otherwise(0)
        )
        #构造observation_time: 如果观测到是正样本，按实际正样本的时间窗口，否则用固定观测窗口 max(1, min(3, attributeWindow - 1))
        # 构造 observation_time
        df = df.withColumn(
            "observation_time",
            F.when(
                col("observed_conversion") == 1,
                col("diff_hours") / 24.0  # 正样本：真实转化时间（天）
            ).otherwise(
                # 负样本：根据 attributeWindow 决定
                F.when(
                    col("attributeWindow") < 3.0,
                    F.greatest(col("attributeWindow") - 1.0, lit(1.0))  # 减1，但不低于1.0
                ).otherwise(
                    lit(3.0)  # attributeWindow >= 3.0 时用 3.0
                )
            )
        )
        
        # 构造 domain_id
        type_to_id_map = self.type_to_id_map
        mapping_expr = F.create_map([F.lit(k) for kv in type_to_id_map.items() for k in kv])
        df = df.withColumn("domain_id",F.coalesce(mapping_expr[F.col("business_type")], F.lit(self.domains_cnt - 1)).cast("int"))
        df = df.withColumn("domain_id", F.lit(0))

        for col_name in df.columns:
            if col_name in ['label', 'domain_id', 'observed_conversion']:
                df = df.withColumn(col_name, F.col(col_name).cast("int"))
            elif col_name in ['observation_time', 'attributeWindow']:
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        df = df.fillna('none')
        return df


if __name__ == "__main__":
    args = DELFModelTrainFlow.parse_args()
    trainer = DELFModelTrainFlow(config_path=args.conf)
    print(f'DELFModelTrainFlow: debug_args= {args}')
    trainer.run_complete_flow(args) 
    MovasLogger.save_to_local()


