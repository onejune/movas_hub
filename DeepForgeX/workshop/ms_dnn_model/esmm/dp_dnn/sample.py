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
                                        worker_count=10,
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

    
# 条件采样逻辑：保留 purchase=1 和 10% 其他样本
def random_sample(df):
    df_filtered = df.filter(
        (F.col("purchase") == 1) |
        (F.rand(seed=42) < 0.1)
    )
    return df_filtered

# 读取某日数据，采样、加标签，并保存采样后文件（若不存在）
def read_or_sample_dataset_by_date(spark_session, base_path, date_str, output_path):
    input_path = os.path.join(base_path, f"sample_{date_str}")
    sampled_path = os.path.join(output_path, f"sampled_{date_str}")

    if os.path.exists(sampled_path):
        print(f"[{date_str}] 已存在采样文件，直接读取: {sampled_path}")
        df = spark_session.read.parquet(sampled_path)
    else:
        print(f"[{date_str}] 正在读取原始数据并进行采样: {input_path}")
        df = spark_session.read.parquet(input_path)
        # df.printSchema()
        # input()
        # 转换字段类型（purchase 为 float，其它为 string）
        for col_name in df.columns:
            if col_name == 'purchase':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            elif col_name == 'diff_hours':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))

        # 筛选
        df = df.filter(F.col("demand_pkgname").isin(["COM.ALIBABA.ALIEXPRESSHD","COM.SS.ANDROID.UGC.TRILL",\
                                                     "COM.SHOPEE", "COM.LAZADA.ANDROID"]))

        # 仅保留 purchase in (0, 1)，填充空值
        df = df.filter(F.col("purchase").isin([0, 1]))
        df = df.fillna('unknown')

        # 添加 dp 标签列
        df = df.withColumn(
            "dp",
            F.when(
                (F.col("diff_hours").cast("float") > 24) & (F.col("purchase") == 1),
                1
            ).otherwise(0)
        )

        # 保存采样结果
        df.write.mode("overwrite").parquet(sampled_path)
        print(f"[{date_str}] 采样并加标签完成，已保存到: {sampled_path}")
    
    return df

# 多天批处理函数（天级递增）
def batch_process_dates(spark_session, base_path, output_path, start_date_str, end_date_str):
    date_format = "%Y-%m-%d"
    current_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)

    while current_date <= end_date:
        date_str = current_date.strftime(date_format)
        try:
            read_or_sample_dataset_by_date(spark_session, base_path, date_str, output_path)
        except Exception as e:
            print(f"[{date_str}] 处理出错: {e}")
        current_date += timedelta(days=1)


if __name__=="__main__":
    print('Debug -- Ctr Demo Wide&Deep')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    # args = parser.parse_args()
    # params = load_config(args.conf)
    # locals().update(params)
    spark = init_spark()
    ## read datasets
    base_path = "/data/walter/ruf/parquet/"         # 原始 daily parquet 文件目录
    output_path = "/data/kailiang/ruf_delay_sample/"      # 采样后带标签的 parquet 输出目录

    batch_process_dates(
        spark,
        base_path,
        output_path,
        start_date_str="2025-02-13",
        end_date_str="2025-02-13"
    )

    print('Debug -- FINISED')
    stop_spark(spark)
