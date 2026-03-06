#cp ../ftrl_maven_walter-0.0.1-SNAPSHOT.jar ./
root_data_path="oss://spark-ml-train-new/liufashuai/dsp/cps/purchase_model/7_train_data"

data_day=$1
data_day=$(date -d "$data_day" +"%Y-%m-%d")
data_path=../sample/sample_${data_day}

model_output="./train_output/base"
#sort -k2,2 -t$'\002' -r -g $model_output > ${model_output}.sort
awk -F'\002' '{if($2!=0) print $0}' $model_output | sort -k2,2 -t$'\002' -r -g > ${model_output}.filt

echo ""
echo "****************************** validation for $data_day ******************************"
time java -Xmx70g -cp ftrl_maven_walter-0.0.1-SNAPSHOT.jar com.mobvista.ftrl.tools.OnlineReplay -data $data_path -conf conf/ -model ${model_output}.filt -out ./train_output/
echo "****************************** validation over *****************************"
echo ""
wc -l ${model_output}.filt
pwd

