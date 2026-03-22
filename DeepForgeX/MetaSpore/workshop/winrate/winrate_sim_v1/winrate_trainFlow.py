# custom_ms_model_train_flow.py
import os, sys, random, traceback
import numpy as np
from typing import List, Any, Optional
from pyspark.sql import functions as F
from dnn_trainFlow import MsModelTrainFlow
import metaspore as ms
from metaspore.algos.deep_censored_model import DeepCensoredModel
from metaspore.algos.widedeep_net import WideDeep
from metaspore.algos.deepfm_net import DeepFM
from metaspore.algos.ffm_net import FFM
from metaspore.algos.dcn_net import DCN
from metaspore.loss_utils import LossUtils
from movas_logger import MovasLogger, how_much_time

class WinrateModelTrainFlow(MsModelTrainFlow):
    """继承MsModelTrainFlow并重载random_sample方法"""
    def __init__(self, config_path):
        super().__init__(config_path)
    
    def _build_model_module(self):
        # 获取模型类型，默认为 WideDeep
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        
        self.model_module1 = WideDeep(
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

        self.model_module = DeepCensoredModel(
            use_wide=self.use_wide,
            batch_norm=self.batch_norm,
            net_dropout=self.net_dropout,
            wide_embedding_dim=self.embedding_size,
            deep_embedding_dim=self.embedding_size,
            wide_combine_schema_path=self.wide_combine_schema_path,
            deep_combine_schema_path=self.combine_schema_path,
            ftrl_l1=self.ftrl_l1,
            ftrl_l2=self.ftrl_l2,
            ftrl_alpha=self.ftrl_alpha,
            ftrl_beta=self.ftrl_beta
        )

    @how_much_time
    def _train_model(self, train_dataset, model_in_path_current, model_out_path_current, model_version_current):
        if not self.model_module:
            self._build_model_module()
            
        estimator = ms.PyTorchEstimator(module=self.model_module,
                                      worker_count=self.worker_count,
                                      server_count=self.server_count,
                                      model_in_path=model_in_path_current,
                                      model_out_path=model_out_path_current,
                                      model_export_path=None, 
                                      model_version=model_version_current,
                                      experiment_name=self.experiment_name,
                                      input_label_column_name='label',
                                      loss_function=LossUtils.censored_regression_loss,
                                      metric_update_interval=1000)
        estimator.updater = ms.AdamTensorUpdater(self.adam_learning_rate)
        
        MovasLogger.add_log(content=f"Starting training for version: {model_version_current}, Model In: {model_in_path_current}, Model Out: {model_out_path_current}")
        try:
            model = estimator.fit(train_dataset)
        except Exception as e:
            traceback.print_exc()
            
        self.trained_model_path = model_out_path_current 
        return model

    def random_sample(self, df):
        df_filtered = df.filter(
            F.rand(seed=42) < 0.01
        )
        return df_filtered

    def _read_dataset_by_date(self, base_path, date_str):
        data_path = os.path.join(base_path, f"{date_str}/*/")
        MovasLogger.add_log(content=f"Reading Parquet data from directory: {data_path}")
        df = self.spark_session.read.parquet(data_path)
        required_columns = self.used_fea_list + ['bidPrice']
        df = df.select(*required_columns)
        df = self.random_sample(df)

        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            elif col_name == 'bidPrice':
                # 专门处理 bidPrice 列的异常值和类型转换
                df = df.withColumn("bidPrice_raw", F.col("bidPrice"))
                
                # 尝试多种方式转换 bidPrice 为数值类型
                df = df.withColumn("bidPrice_numeric",
                                F.when(
                                    # 如果已经是数值类型
                                    F.col("bidPrice").rlike(r"^[0-9]+\.?[0-9]*$"),
                                    F.col("bidPrice").cast("float")
                                ).otherwise(F.lit(None)))
                # 过滤掉无效的 bidPrice（NaN、负值、过大值）
                df = df.filter(
                    (F.col("bidPrice_numeric").isNotNull()) & 
                    (F.col("bidPrice_numeric") >= 0) &
                    (F.col("bidPrice_numeric") <= 100)  # 设置合理的上限
                )
                # 将处理后的数值列重命名为 bidPrice
                df = df.withColumn("bidPrice", F.col("bidPrice_numeric")) \
                    .drop("bidPrice_numeric", "bidPrice_raw")  # 清理临时列
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        #df = df.filter(F.col("label").isin([0, 1]))
        
        # 对 string 列填充 'none'
        df = df.fillna('none')  
        # 特别确保 bidPrice 是数值类型（如果仍有空值则设为 0）
        df = df.withColumn("bidPrice", F.coalesce(F.col("bidPrice").cast("float"), F.lit(0.0)))
        MovasLogger.add_log(content=f"Reading Parquet over!!!")
        return df

if __name__ == "__main__":
    args = WinrateModelTrainFlow.parse_args()
    trainer = WinrateModelTrainFlow(config_path=args.conf)
    try:
        trainer.run_complete_flow(args) 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    MovasLogger.save_to_local()


