import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum
from datetime import datetime, timedelta

def count_labels_per_day(spark, input_base_path, start_day, end_day):
    date_format = "%Y-%m-%d"
    cur = datetime.strptime(start_day, date_format)
    end = datetime.strptime(end_day, date_format)

    print("日期 | purchase=1 数量 | ip=1 数量 | dp=1 数量")
    print("正例|立即正例|延迟正例")
    print("-" * 50)

    while cur <= end:
        day_str = cur.strftime(date_format)
        input_path = os.path.join(input_base_path, f"sampled_{day_str}")

        if not os.path.exists(input_path):
            print(f"{day_str} | 文件不存在，跳过")
            cur += timedelta(days=1)
            continue

        try:
            df = spark.read.parquet(input_path)

            stats = df.agg(
                # spark_sum(col("real_purchase")).alias("real_purchase_1_cnt"),
                spark_sum(col("purchase")).alias("purchase_1_cnt"),
                spark_sum(col("dp")).alias("dp_1_cnt")
            ).collect()[0]

            # print(f"{day_str} | purchase={int(stats['real_purchase_1_cnt'])} | im={int(stats['purchase_1_cnt'])} | dp={int(stats['dp_1_cnt'])}")
            print(f"{day_str} | purchase={int(stats['purchase_1_cnt'])} | dp={int(stats['dp_1_cnt'])}")
        except Exception as e:
            df = spark.read.parquet(input_path)
            df.printSchema()
            # print(f"{day_str} | 读取失败: {str(e)}")

        cur += timedelta(days=1)

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("CountDailyPurchaseStats") \
        .config("spark.local.dir", "/data/spark/tmp") \
        .getOrCreate()

    input_base_path = "/data/kailiang/ruf2_delay_sample_renamed"  # parquet 重命名后的目录
    count_labels_per_day(
        spark,
        input_base_path,
        start_day="2025-04-01",
        end_day="2025-04-10"
    )

    spark.stop()
