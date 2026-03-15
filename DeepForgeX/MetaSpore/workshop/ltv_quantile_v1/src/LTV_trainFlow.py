# custom_ms_model_train_flow.py
import os, sys, random, traceback
import numpy as np
import shutil, inspect, math
from operator import itemgetter
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
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
from metaspore.algos.ziln_model import ZILNModel
from metaspore.algos.widedeep_net import WideDeep
from metaspore.algos.deepfm_net import DeepFM
from metaspore.algos.ffm_net import FFM
from metaspore.algos.dcn_net import DCN
from metaspore.algos.ltv import LTVQuantileRegressionModel
from metaspore.loss_utils import get_loss_function
from movas_logger import MovasLogger, how_much_time

class LtvModelTrainFlow(MsModelTrainFlow):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.output_csv_path = '/mnt/data/oss_wanjun/01_project/dnn_experiment/output_report/ltv_exp_report.csv'
        self.batch_days = 30
    
    def _build_model_module(self):
        # 获取模型类型，默认为 WideDeep
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        if configed_model == "WideDeep":
            self.model_module = WideDeep(
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
                    ftrl_beta=self.ftrl_beta,
                    final_net_activation=None #该参数用于控制是否激活最后一层（sigmoid），回归任务不需要激活
                )
        elif configed_model == "quantile":
            self.model_module = LTVQuantileRegressionModel(
                    batch_norm=self.batch_norm,
                    net_dropout=self.net_dropout,
                    sparse_embedding_dim=self.embedding_size,
                    combine_schema_path=self.wide_combine_schema_path,
                    hidden_units=self.dnn_hidden_units,
                    hidden_activations="relu",
                    ftrl_l1=self.ftrl_l1,
                    ftrl_l2=self.ftrl_l2,
                    ftrl_alpha=self.ftrl_alpha,
                    ftrl_beta=self.ftrl_beta,
                    target_quantile=0.7
                )
        elif configed_model == "ZilnModel":
            self.model_module = ZILNModel(
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

    @how_much_time
    def _train_model(self, train_dataset, model_in_path_current, model_out_path_current, model_version_current):
        if not self.model_module:
            self._build_model_module()
        loss_func = get_loss_function(self.loss_func)
        if not loss_func:
            raise ValueError("Invalid loss function specified.")
            
        estimator = ms.PyTorchEstimator(module=self.model_module,
                                      worker_count=self.worker_count,
                                      server_count=self.server_count,
                                      model_in_path=model_in_path_current,
                                      model_out_path=model_out_path_current,
                                      model_export_path=None, 
                                      model_version=model_version_current,
                                      experiment_name=self.experiment_name,
                                      input_label_column_name='label',
                                      loss_function=loss_func,
                                      loss_type='regression', 
                                      metric_update_interval=1000,
                                      task_to_id_map = self.task_to_id_map,
                                      task_weights_dict = self.task_weights_dict)
        estimator.updater = ms.AdamTensorUpdater(self.adam_learning_rate)
        
        MovasLogger.add_log(content=f"Starting training for version: {model_version_current},  \
                Model In: {model_in_path_current}, Model Out: {model_out_path_current}, \
                model_type: {self.model_type}, \
                configed_loss_func: {self.loss_func}, \
                loss_func: {loss_func}")
        model = estimator.fit(train_dataset)
            
        self.trained_model_path = model_out_path_current 
        return model

    def _preprocess_dataset(self, df):
        required_columns = self.used_fea_list + ['revenue']
        required_columns.remove('label')
        df = df.select(*required_columns)

        for col_name in df.columns:
            if col_name == 'revenue':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))

        #df = self.random_sample(df)
        df = df.filter("business_type not in('shein', 'aerta', 'lazada_rta') and revenue>0 and revenue<100")
        target = 0.0012345
        epsilon = 1e-7  # 根据你的数据精度调整，比如 0.0000001

        # 过滤掉 revenue 接近 target 的行
        df = df.filter(
            F.abs(F.col('revenue') - target) > epsilon
        )
        df = df.withColumn('label', F.col('revenue'))
        df = df.fillna('none')
        MovasLogger.add_log(content=f"Reading Parquet over!!! sample count: {df.count()}")
        return df

    def _eval_by_df(self, col_name, col_value, filtered_df, result_dict):
        EvalResult = namedtuple('EvalResult', ['key1', 'key2', 'auc', 'pcoc', 'loss', 'pos', 'neg', 'ivr'])
        # 提取 (label, rawPrediction) 列并收集到 Driver 端
        if self.loss_func != 'wce_loss':
            label_pred_list = (
                filtered_df.select("label", "rawPrediction")
                .rdd.map(lambda row: (float(row.label), float(row.rawPrediction)))
                .collect()
            )
        else: #wce loss需要对 forward 的结果进行幂运算
            label_pred_list = (
                filtered_df.select("label", "rawPrediction")
                .rdd.map(lambda row: (float(row.label), math.exp(float(row.rawPrediction))))
                .collect()
            )
        # 初始化评估指标和样本数
        auc, pcoc, logloss = 0.0, 0.0, 0.0
        sample_count, avg_label = 0, 0
        
        if label_pred_list:
            # 计算 AUC 和 PCOC
            auc, pcoc = compute_auc_and_pcoc_regression(label_pred_list)
            sample_count = len(label_pred_list)
            avg_label = sum(label for label, _ in label_pred_list) / sample_count
        
        #print(f"{col_name} {col_value} auc: {auc}, pcoc: {pcoc}, loss: {logloss}, pos: {sample_count}, neg: {sample_count}, ivr: {avg_label}")
        # 使用命名元组存储结果
        result_dict[col_value] = EvalResult(
            key1=col_name,
            key2=col_value,
            auc=auc,
            pcoc=pcoc,
            loss=logloss,
            pos=sample_count,
            neg=sample_count,
            ivr=avg_label
        )
    @how_much_time
    def _predict_data(self, dataset_to_transform, model_in_path_current):
        if not self.model_module:
            self._build_model_module()

        model_transformer = ms.PyTorchModel(module=self.model_module,
                                worker_count=self.worker_count,
                                server_count=self.server_count,
                                model_in_path=model_in_path_current, 
                                experiment_name=self.experiment_name,
                                input_label_column_name='label',
                                loss_type='regression', #回归模型在Validation 的时候需要加上这个参数
                                task_to_id_map = self.task_to_id_map,
                                task_weights_dict = self.task_weights_dict)

        MovasLogger.add_log(content=f"Transforming data using model from: {model_in_path_current}")
        test_result = model_transformer.transform(dataset_to_transform)
        #MovasLogger.add_log(content=f"Test result sample:\n{MovasLogger.get_df_showString(test_result, lines=5)}")
        return test_result


if __name__ == "__main__":
    args = LtvModelTrainFlow.parse_args()
    trainer = LtvModelTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args) 
    MovasLogger.save_to_local()


