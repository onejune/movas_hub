# multi-task learning framework

import os, sys, random, traceback, json
import numpy as np
import shutil, inspect, math
from operator import itemgetter
from pyspark.sql.types import StringType,ArrayType, IntegerType, StructField, StructType, MapType
from pyspark.sql import functions as F
from pyspark.sql.functions import create_map, coalesce, lit
from pyspark.sql.functions import rand, to_json, col
from datetime import datetime, timedelta
import time
from tabulate import tabulate
from collections import namedtuple
from typing import List, Any, Optional
from .base import BaseTrainFlow
import metaspore as ms

from metaspore.algos.multi_task import MtlMMoEModel, MdlMMoEModel, MMoE, MtlSharedBottomModel, SceneAwareMMoE, HoME, MultiLayerPLEMD, PEPNet, PEPNet2, STEMNet, M2MModel
from metaspore.algos.multi_domain import HMoENetwork, STARModel
from metaspore.algos.widedeep_net import WideDeep
from metaspore.loss_utils import get_loss_function
from metaspore.loss_utils import LossUtils

from metaspore.utils import MovasLogger, how_much_time

MovasLogger.set_debug_mode(False)

class MTLTrainFlow(BaseTrainFlow):
    def __init__(self, config_path):
        super().__init__(config_path)
    
    def _build_model_module(self):
        configed_model = self.params.get('model_type')
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        if configed_model == "mmoe_mtl":
            self.model_module = MtlMMoEModel(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                expert_numb=3,
                task_numb=self.tasks_cnt,
                expert_out_dim=8,
                expert_hidden_units=self.dnn_hidden_units,
                tower_hidden_units=[64],
                gate_hidden_units=[64],
                batch_norm=False,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta,
                use_uncertainty_weighting=self.use_uncertainty_weighting,
                use_gradnorm=self.use_gradnorm,
                gradnorm_alpha=self.gradnorm_alpha,
                gate_l2_reg_lambda=self.gate_l2_reg_lambda,
                expert_dissim_reg_lambda=self.expert_dissim_reg_lambda,
                expert_f_norm_reg_lambda=self.expert_f_norm_reg_lambda,
                load_balancing_reg_lambda=self.load_balancing_reg_lambda,
                temperature=1.0
            )
        elif configed_model == "m2m":
            self.model_module = M2MModel(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                scene_combine_schema_path=self.scene_combine_schema_path,
                ad_combine_schema_path=self.ad_combine_schema_path,
                num_experts=2,
                num_tasks=self.tasks_cnt,
                expert_dim=64,
                scenario_dim=64,
                meta_hidden_dim=64,
                expert_hidden_units=self.dnn_hidden_units,
                gate_hidden_units = [128, 64],
                tower_hidden_units = [128, 64],
                batch_norm=False,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "sa_mmoe": # 场景感知MMoE
            self.model_module = SceneAwareMMoE(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                scene_combine_schema_path=self.scene_combine_schema_path,
                expert_numb=3,
                task_numb=self.tasks_cnt,
                expert_out_dim=8,
                expert_hidden_units=self.dnn_hidden_units,
                tower_hidden_units=[32],
                gate_hidden_units=[32],
                batch_norm=False,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "ple": # PLE
            self.model_module = MultiLayerPLEMD(
                num_layers=2,
                domain_count=1,
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                shared_expert_num=3,
                domain_expert_num=1,
                task_count=self.tasks_cnt,
                expert_hidden_units=self.dnn_hidden_units,
                task_towers_hidden_units=[32],
                gate_hidden_units=[32],
                batch_norm=False,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta,
                use_uncertainty_weighting=self.use_uncertainty_weighting
            )
        elif configed_model == "home":
            self.model_module = HoME(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                task_numb=self.tasks_cnt,
                task_groups=[[0,1], [2,3]],
                meta_expert_hidden_units=self.dnn_hidden_units,
                meta_gate_hidden_units=[32],
                expert_hidden_units=[64],
                expert_out_dim=8,
                gate_hidden_units=[32],
                tower_hidden_units=[32],
                batch_norm=True,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "shared_bottom":
            self.model_module = MtlSharedBottomModel(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                task_numb=self.tasks_cnt,
                bottom_hidden_units=self.dnn_hidden_units,
                tower_hidden_units=[64],
                batch_norm=False,
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
                prior_combine_schema_path=self.ppnet_combine_schema_path,
                task_towers_hidden_units=self.dnn_hidden_units,
                task_count=self.tasks_cnt,
                domain_count=1,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta,
                use_uncertainty_weighting=self.use_uncertainty_weighting
            )
        elif configed_model == "pepnet2": #with shared MLP bottom
            self.model_module = PEPNet2(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                prior_combine_schema_path=self.ppnet_combine_schema_path,
                base_dnn_hidden_units=self.dnn_hidden_units,
                task_towers_hidden_units=self.task_tower_hidden_units,
                task_count=self.tasks_cnt,
                domain_count=1,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta,
                use_uncertainty_weighting=self.use_uncertainty_weighting
            )
        elif configed_model == "hmoe":
            self.model_module = HMoENetwork(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                domain_count=self.domains_cnt,
                task_count=self.tasks_cnt,
                expert_units=self.dnn_hidden_units,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "stem": 
            self.model_module = STEMNet(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                shared_hidden_units=self.dnn_hidden_units,
                tower_hidden_units=self.task_tower_hidden_units,
                task_count=self.tasks_cnt,
                batch_norm=self.batch_norm,
                dropout_rate=self.net_dropout,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta,
                use_uncertainty_weighting=self.use_uncertainty_weighting
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

    def add_random_event_column(self, df):
        # 定义在函数外部或使用独立函数
        def generate_random_event():
            import random  # 显式导入，避免依赖外部 random
            import json
            possible_events = ['purchase', 'atc', 'open', 'content_view']
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

        used_fea_list = self.used_fea_list + ['mul_labels']
        df = df.select(*used_fea_list)

        df = self.random_sample(df)
        MovasLogger.log(f"final_sample_cnt: {df.count()}")

        df = df.withColumn("mul_labels", to_json(col("mul_labels")))
        
        # 构造 domain_id
        type_to_id_map = self.type_to_id_map
        mapping_expr = F.create_map([F.lit(k) for kv in type_to_id_map.items() for k in kv])
        df = df.withColumn("domain_id",F.coalesce(mapping_expr[F.col("business_type")], F.lit(self.domains_cnt - 1)).cast("int"))
        
        df = df.withColumn("domain_id", F.lit(0))

        for col_name in df.columns:
            if col_name in ['label', 'domain_id']:
                df = df.withColumn(col_name, F.col(col_name).cast("int"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        df = df.fillna('none')
        return df


if __name__ == "__main__":
    args = MTLTrainFlow.parse_args()
    trainer = MTLTrainFlow(config_path=args.conf)
    print(f'MTLTrainFlow: debug_args= {args}')
    trainer.run_complete_flow(args) 
    MovasLogger.save_to_local()