import sys
import os
sys.path.insert(0, './src')
from dnn_trainFlow import DNNModelTrainFlow
from pyspark.sql import functions as F

flow = DNNModelTrainFlow('./conf/widedeep.yaml')
flow._init_spark()

# 读取数据
sample_date = "2026-03-03"
model_date = "2026-03-02"
model_path = f"./output/model_{model_date}"

print(f"读取数据: {sample_date}")
test_dataset = flow._read_dataset_by_date(flow.train_path_prefix, sample_date)

print(f"加载模型并预测: {model_path}")
test_result_df = flow._predict_data(test_dataset, model_path)

# 检查列名
print(f"可用列: {test_result_df.columns[:20]}...")

# 保存预测结果 - 使用 rawPrediction 而不是 prediction
pred_output_path = f"./output/predictions_{sample_date}"

# 找到预测列
pred_col = "rawPrediction" if "rawPrediction" in test_result_df.columns else "prediction"
print(f"使用预测列: {pred_col}")

save_cols = ["label", pred_col]
key_features = ["country", "adx", "demand_pkgname", "business_type", "devicetype", "os"]
for col in key_features:
    if col in test_result_df.columns and col not in save_cols:
        save_cols.append(col)

print(f"保存预测结果到: {pred_output_path}")
print(f"保存列: {save_cols}")

# 重命名 rawPrediction 为 prediction
if pred_col == "rawPrediction":
    test_result_df = test_result_df.withColumn("prediction", F.col("rawPrediction"))
    save_cols = ["label", "prediction"] + [c for c in save_cols if c not in ["label", "rawPrediction"]]

test_result_df.select(*save_cols).write.mode("overwrite").parquet(pred_output_path)
print("保存完成!")

flow._stop_spark()
