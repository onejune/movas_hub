#$1:model date
#$2:train data date
root_data_path="oss://spark-ml-train-new/liufashuai/dsp/cps/purchase_model/7_train_data"
model_v1="oss://spark-ml-train-new/liufashuai/dsp/cps/purchase_model/8_lr_model"
model_v1="oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_purchase/lr_model_v3"
model_v1="oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_purchase/lr_huf_model_v2"
model_v1="oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_model/self_attr_model_v1"
model_v1="oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_model/self_attr_duf_inner_v2"

model_path=$model_v1
model_dtm=$1
model_output="./train_output/model_file_$model_dtm"
data_day=$2
data_path=../sample/sample_${data_day}

#cat $data_path | grep "DUF_INNER_V2" > ./temp/sample_$data_day
#data_path="./temp/sample_$data_day"

if [[ ! -f "${model_output}.filt" ]]; then
	#download model
	model_oss_path=$model_path/model_file_${model_dtm}
	ossutil cp $model_oss_path $model_output
	awk -F'\002' '{if($2!=0) print $0}' $model_output | sort -k2,2 -t$'\002' -r -g > ${model_output}.filt
fi

echo ""
echo "****************************** validation: model=$model_output, val_date=$data_day ******************************"
time java -Xmx70g -cp ftrl_maven_walter-0.0.1-SNAPSHOT.jar com.mobvista.ftrl.tools.OnlineReplay -data $data_path -conf conf/ -model ${model_output}.filt -out ./train_output/ -name "val-$data_day"
echo "****************************** validation over: model=$model_output, val_date=$data_day *****************************"

echo ""
wc -l ${model_output}.filt
pwd

