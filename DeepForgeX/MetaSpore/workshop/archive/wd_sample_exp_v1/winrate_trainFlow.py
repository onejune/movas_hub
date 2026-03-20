# custom_ms_model_train_flow.py
import os, sys, random
import numpy as np
from typing import List, Any, Optional
from pyspark.sql import functions as F
from dnn_trainFlow import MsModelTrainFlow
from movas_logger import MovasLogger, how_much_time

class WinrateModelTrainFlow(MsModelTrainFlow):
    """继承MsModelTrainFlow并重载random_sample方法"""
    def __init__(self, config_path):
        super().__init__(config_path)
        
    def random_sample(self, df):
        df_filtered = df.filter(
            F.rand(seed=42) < 0.05
        )
        return df_filtered

     #@how_much_time
    def _read_dataset_by_date(self, base_path, date_str):
        data_path = os.path.join(base_path, f"{date_str}/*/")
        MovasLogger.add_log(content=f"Reading Parquet data from directory: {data_path}")
        df = self.spark_session.read.parquet(data_path)
        df = df.select(*self.used_fea_list) #important：用 combine schema 过滤一遍 col，优化性能
        
        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        df = self.random_sample(df)

        #df = df.filter(F.col("label").isin([0, 1]))
        df = df.fillna('unknown') #important: 会报错！
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


