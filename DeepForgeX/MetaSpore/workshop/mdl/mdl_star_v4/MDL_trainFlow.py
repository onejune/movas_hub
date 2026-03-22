# multi-domain learning framework

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
from metaspore.algos.star_gate_net import StarGateModel

from metaspore.algos.pepnet_net import PEPNet
from metaspore.algos.multitask import MtlMMoEModel, MdlMMoEModel, MMoE
from metaspore.algos.widedeep_net import WideDeep
from metaspore.loss_utils import get_loss_function
from movas_logger import MovasLogger, how_much_time

class MDLModelTrainFlow(MsModelTrainFlow):
    def __init__(self, config_path):
        super().__init__(config_path)
    
    def _build_model_module(self):
        # 获取模型类型，默认为 WideDeep
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        if configed_model == "star":
            self.model_module = STARModel(
                    num_domains = self.domains_cnt,
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
        elif configed_model == "star_gate":
            self.model_module = StarGateModel(
                    num_domains = self.domains_cnt,
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
        elif configed_model == "mmoe_mdl":
            self.model_module = MdlMMoEModel(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                expert_numb=4,
                domain_numb=self.domains_cnt,
                expert_out_dim=10,
                expert_hidden_units=self.dnn_hidden_units,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "pepnet":
            self.model_module = PEPNet(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                ppnet_combine_schema_path=self.ppnet_combine_schema_path,
                task_towers_hidden_units=self.dnn_hidden_units,
                task_count=1, #pepnet本身是出来多任务多场景
                domain_count=self.domains_cnt,
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

    def _preprocess(self):
        self.type_to_id_map = {
            'shein': 0,
            'shopee_cps': 1,
            'aecps': 2,
            'aerta': 3,
            'everglowly': 4,
            'lazada_rta': 3,
            'lazada_cps': 3,
            'ttshop': 4,
            'miravia_rta': 4,
            'liongame': 4,
            'cyberclickshecurve': 4,
            'detroitrain': 4,
            'swiftlink': 4,
            'saker': 4,
            'starpony': 4,
            'mena_ae_cps': 3
        }
        domains_cnt = len(list(set(self.type_to_id_map.values())))
        MovasLogger.log(f'type_to_id_map = {self.type_to_id_map}, domains_cnt = {domains_cnt}')
        self.domains_cnt = domains_cnt + 1

    def _read_dataset_by_date(self, base_path, date_str):
        data_path = os.path.join(base_path, f"part={date_str}")
        df = self.spark_session.read.parquet(data_path)
        df = df.select(*self.used_fea_list) 
        df = self.random_sample(df)

        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        
        # 使用 Spark SQL 表达式构建 domain_id，避免 UDF
        # 构造 map literal
        mapping_expr = F.create_map([F.lit(k) for kv in self.type_to_id_map.items() for k in kv])
        df = df.withColumn(
                "domain_id",
                F.coalesce(mapping_expr[F.col("business_type")], F.lit(self.domains_cnt - 1)).cast("int")
            )
        df = df.withColumn("task_id", F.lit(0)) # 默认只有 1 个 task
        df = df.fillna('none')
        df.select("business_type", "domain_id", "task_id").show(truncate=False)
        return df


if __name__ == "__main__":
    args = MDLModelTrainFlow.parse_args()
    trainer = MDLModelTrainFlow(config_path=args.conf)
    print(f'MDLModelTrainFlow: debug_args= {args}')
    trainer.run_complete_flow(args) 
    MovasLogger.save_to_local()


