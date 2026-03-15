# multi-task learning framework

import os, sys, random, traceback, json
import numpy as np
import shutil, inspect, math
from operator import itemgetter
from pyspark.sql.types import StringType,ArrayType, IntegerType, StructField, StructType, MapType
from pyspark.sql import functions as F
from pyspark.sql.functions import create_map, coalesce, lit
from metrics_eval import compute_auc_and_pcoc_regression
from pyspark.sql.functions import rand
from datetime import datetime, timedelta
import time
from tabulate import tabulate
from collections import namedtuple
from typing import List, Any, Optional
from dnn_trainFlow import MsModelTrainFlow
import metaspore as ms
from metaspore.algos.star_net import STARModel
from metaspore.algos.pepnet_net import PEPNet
from metaspore.algos.multitask import MtlMMoEModel, MdlMMoEModel, MMoE
from metaspore.algos.widedeep_net import WideDeep
from metaspore.loss_utils import get_loss_function
from movas_logger import MovasLogger, how_much_time

MovasLogger.set_debug_mode(True)

class MTLModelTrainFlow(MsModelTrainFlow):
    def __init__(self, config_path):
        super().__init__(config_path)
    
    def _build_model_module(self):
        # 获取模型类型，默认为 WideDeep
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        if configed_model == "mmoe11111":
            self.model_module = MtlMMoEModel(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                expert_numb=4,
                task_numb=5,
                expert_out_dim=10,
                expert_hidden_units=self.dnn_hidden_units,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "mmoe":
            self.model_module = MdlMMoEModel(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                expert_numb=4,
                domain_numb=5,
                expert_out_dim=10,
                expert_hidden_units=self.dnn_hidden_units,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
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
                    F.col("business_type") == 'shein',
                    F.rand(seed=42) < 0.01  # shein负样本采样1%
                )
                .otherwise(
                    F.rand(seed=42) < 0.01   # 其他负样本采样10%
                )
            )
        )
        return df_filtered

    def add_random_event_column(self, df):
        # 定义在函数外部或使用独立函数
        def generate_random_event():
            import random  # 显式导入，避免依赖外部 random
            import json
            possible_events = ['purchase', 'atc', 'checkout', 'open', 'content_view']
            k = random.randint(2, 4)
            selected_events = random.sample(possible_events, k)
            event_dict = {e: random.choice([0, 1]) for e in selected_events}
            return json.dumps(event_dict)

        generate_event_udf = F.udf(generate_random_event, StringType())
        return df.withColumn("event", generate_event_udf())

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
            'purchase': 1,
            'atc': 2,
            'checkout': 3,
            'open': 4,
            'content_view': 5
        }

    def _read_dataset_by_date(self, base_path, date_str):
        data_path = os.path.join(base_path, f"part={date_str}")
        df = self.spark_session.read.parquet(data_path)
        df = df.select(*self.used_fea_list) 
        df = self.random_sample(df)
        MovasLogger.log(f"before_sample_cnt: {df.count()}")
        df = self.add_random_event_column(df)
        
        # 构造 domain_id
        type_to_id_map = self.type_to_id_map
        mapping_expr = F.create_map([F.lit(k) for kv in type_to_id_map.items() for k in kv])
        df = df.withColumn("domain_id",F.coalesce(mapping_expr[F.col("business_type")], F.lit(self.domains_cnt - 1)).cast("int"))

        # 按 (task, label) 展开样本
        result_df = self.expand_events_to_samples(df)
        result_df.select("business_type", "domain_id", "event", "task_id", "label").show(truncate=False, n=10)
        MovasLogger.log(f"final_sample_cnt: {result_df.count()}")

        for col_name in result_df.columns:
            if col_name in ['label', 'task_id']:
                result_df = result_df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                result_df = result_df.withColumn(col_name, F.col(col_name).cast("string"))
        result_df = result_df.fillna('none')
        
        return result_df

    def expand_events_to_samples(self, df):
        """
        根据event字段对样本进行explode，使每条样本对应一个task_id和label
        """
        # 解析event JSON字符串为map类型
        def parse_event_json(json_str):
            if json_str:
                try:
                    return json.loads(json_str)
                except:
                    return {}
            return {}
        
        parse_event_udf = F.udf(parse_event_json, MapType(StringType(), IntegerType()))
        df_parsed = df.withColumn("event_parsed", parse_event_udf(F.col("event")))
        
        # 获取原始列名（排除临时列）
        original_cols = [col_name for col_name in df.columns if col_name != 'event_parsed']
        
        # 将map展开为多行，使用transform和flatten来避免复杂的posexplode操作
        # 首先将map转换为结构体数组
        df_with_array = df_parsed.withColumn(
            "event_entries",
            F.expr("""
                transform(map_keys(event_parsed), key -> struct(key as task_name, element_at(event_parsed, key) as task_label))
            """)
        )
        
        # 展开数组
        df_exploded = df_with_array.select(
            *[col for col in original_cols],
            F.explode("event_entries").alias("event_entry")
        ).select(
            *[col for col in original_cols],
            F.col("event_entry.task_name").alias("task_name"),
            F.col("event_entry.task_label").alias("task_label")
        )
        
        # 创建任务名称到ID的映射
        task_keys = []
        task_values = []
        for k, v in self.task_to_id_map.items():
            task_keys.append(F.lit(k))
            task_values.append(F.lit(v))
        task_mapping_expr = F.create_map([item for pair in zip(task_keys, task_values) for item in pair])
        
        # 最终结果，包含原始列、task_id和label
        result_df = df_exploded.select(
            *[col_name for col_name in original_cols if col_name not in ['task_id', 'label']],
            F.coalesce(task_mapping_expr[F.col("task_name")], F.lit(0)).alias("task_id"),
            F.col("task_label").alias("label")
        )
        
        return result_df


if __name__ == "__main__":
    args = MTLModelTrainFlow.parse_args()
    trainer = MTLModelTrainFlow(config_path=args.conf)
    print(f'MTLModelTrainFlow: debug_args= {args}')
    try:
        trainer.run_complete_flow(args) 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    MovasLogger.save_to_local()


