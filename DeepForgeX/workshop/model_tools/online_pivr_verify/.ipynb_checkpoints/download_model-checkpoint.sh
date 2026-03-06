model_path="oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_purchase/lr_huf_model_v2"
model_path="oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ruf_sample_v1/model_ruf_v1"

model_dtm=$1
model_oss_path=$model_path/online_model_${model_dtm}
ossutil cp $model_oss_path ./data/
