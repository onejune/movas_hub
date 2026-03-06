#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import metaspore as ms
import pyspark
import numpy as np
import yaml
import subprocess
import argparse
import sys 
from operator import itemgetter
import os
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
import datetime
from datetime import datetime, timedelta


def load_config(path):
    params=dict()
    with open(path,'r') as stream:
        params=yaml.load(stream,Loader=yaml.FullLoader)
        print('Debug--load config:',params)
    return params

def init_spark():
    # subprocess.run(['zip', '-r', 'demo/ctr/widedeep/python.zip', 'python'], cwd='../../../')
    spark_confs={
        "spark.network.timeout":"500",
        "spark.local.dir": "/data/spark/tmp", 
    }
    spark_session = ms.spark.get_session(local=True,
                                        app_name="Sample",
                                        batch_size=256,
                                        worker_count=20,
                                        server_count=2,
                                        worker_memory='10G',
                                        server_memory='10G',
                                        coordinator_memory='10G',
                                        spark_confs=spark_confs)
    
    sc = spark_session.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark_session

def stop_spark(spark):
    print('Debug--spark stop')
    spark.sparkContext.stop()

    
def build_down_sample_for_day(spark_session, base_path, output_path, day_str, window_days=7):
    """
    为指定曝光日（如 '20250511'）构建样本。
    """

    date_format = "%Y-%m-%d"
    path = os.path.join(base_path, f"sampled_{day_str}")
    if os.path.exists(path):
        print(f"读取并筛选: {path}")
        df = spark_session.read.parquet(path)
        df_filtered = df.filter(
            (F.col("purchase") == 1) |
            (F.rand(seed=42) < 0.1)
        )
        # shuffle：局部打乱（避免全局排序带来的高内存压力）
        df_filtered = df_filtered.repartition(200)

        # 写出
        output_dir = os.path.join(output_path, f"sampled_{day_str}")
        df_filtered.write.mode("overwrite").parquet(output_dir)

        print(f"[{day_str}] 样本生成完成，保存于: {output_dir}")
    else:
        print(f"跳过（文件不存在）: {path}")
        return
    
    

def batch_build_down_samples(spark_session, base_path, output_path, start_day, end_day):
    date_format = "%Y-%m-%d"
    cur = datetime.strptime(start_day, date_format)
    end = datetime.strptime(end_day, date_format)

    while cur <= end:
        day_str = cur.strftime(date_format)
        build_down_sample_for_day(spark_session, base_path, output_path, day_str)
        cur += timedelta(days=1)



if __name__=="__main__":
    print('Debug -- Ctr Demo Wide&Deep')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    # args = parser.parse_args()
    # params = load_config(args.conf)
    # locals().update(params)
    spark = init_spark()
    ## read datasets
    base_path = "/data/kailiang/ruf2_delay_sample/"         # 原始 daily parquet 文件目录
    output_path = "/data/kailiang/ruf2_delay_down_sample/"      # 采样后带标签的 parquet 输出目录

    batch_build_down_samples(
        spark,
        base_path,
        output_path,
        start_day="2025-04-11",
        end_day="2025-07-10"
    )

    print('Debug -- FINISED')
    stop_spark(spark)
