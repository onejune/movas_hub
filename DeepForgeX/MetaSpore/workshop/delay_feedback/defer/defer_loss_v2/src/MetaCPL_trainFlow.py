#Meta-Conversion Path Learning (MetaCPL)
#Adaptive Conversion Path Network (ACPN)

# custom_ms_model_train_flow.py
import os, sys, random, traceback
import numpy as np
import shutil, inspect, math
from operator import itemgetter
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import create_map, coalesce, lit
from metrics_eval import compute_auc_and_pcoc_regression
from pyspark.sql.functions import rand
from datetime import datetime, timedelta
import time
from tabulate import tabulate
from collections import namedtuple
from typing import List, Any, Optional
from pyspark.sql import functions as F
from dnn_trainFlow import MsModelTrainFlow
import metaspore as ms
from metaspore.algos.star_net import STARModel
from metaspore.algos.multitask import MMoEProModel
from metaspore.algos.widedeep_net import WideDeep
from metaspore.loss_utils import get_loss_function
from movas_logger import MovasLogger, how_much_time

class MetaCPLTrainFlow(MsModelTrainFlow):
    def __init__(self, config_path):
        super().__init__(config_path)
    
    def _build_model_module(self):
        # 获取模型类型，默认为 WideDeep
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        if configed_model == "star":
            self.model_module = STARModel(
                    num_domains = 20,
                    center_hidden_units = self.dnn_hidden_units,
                    batch_norm=self.batch_norm,
                    net_dropout=self.net_dropout,
                    embedding_dim=self.embedding_size,
                    combine_schema_path=self.combine_schema_path,
                    ftrl_l1=self.ftrl_l1,
                    ftrl_l2=self.ftrl_l2,
                    ftrl_alpha=self.ftrl_alpha,
                    ftrl_beta=self.ftrl_beta
                )
        elif configed_model == "mmoe_pro":
            self.model_module = MMoEProModel(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                expert_numb=6,
                domain_numb=20,
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
                    F.rand(seed=42) < 0.1   # 其他负样本采样10%
                )
            )
        )
        return df_filtered
    def _read_dataset_by_date(self, base_path, date_str):
        type_to_id_map = {
            'shein': 0,
            'shopee_cps': 1,
            'aecps': 2,
            'aerta': 3,
            'everglowly': 4,
            'lazada_rta': 5,
            'lazada_cps': 6,
            'ttshop': 7,
            'miravia_rta': 5,
            'liongame': 4,
            'cyberclickshecurve': 10,
            'detroitrain': 4,
            'swiftlink': 4,
            'saker': 4,
            'starpony': 14,
            'alibaba': 15,
            'mena_ae_cps': 16
        }
        domains_cnt = len(type_to_id_map.keys())
        print(f'type_to_id_map = {type_to_id_map}')

        data_path = os.path.join(base_path, f"part={date_str}")
        df = self.spark_session.read.parquet(data_path)
        df = df.select(*self.used_fea_list) 
        
        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        
        # 使用 Spark SQL 表达式构建 domain_id，避免 UDF
        # 构造 map literal
        mapping_expr = create_map([lit(k) for kv in type_to_id_map.items() for k in kv])
        df = df.withColumn(
                "domain_id",
                coalesce(mapping_expr[F.col("business_type")], F.lit(domains_cnt + 1)).cast("int")
            )
        df = self.random_sample(df)
        df = df.fillna('none')
        #df = df.repartition(50)  # 减少分区数，让每个分区更大
        #MovasLogger.log(f'debug_sample_count= {df.count()}')
        df.select("business_type", "domain_id").show()
        return df


if __name__ == "__main__":
    args = MetaCPLTrainFlow.parse_args()
    trainer = MetaCPLTrainFlow(config_path=args.conf)
    print(f'MetaCPLTrainFlow: debug_args= {args}')
    try:
        trainer.run_complete_flow(args) 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    MovasLogger.save_to_local()


