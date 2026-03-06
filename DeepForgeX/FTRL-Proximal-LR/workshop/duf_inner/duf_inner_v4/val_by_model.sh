model_output="./train_output/base.$1"
data_day=$2
data_path=../sample/sample_${data_day}
#data_path=../sample/sample_${data_day}

if [[ ! -f "${model_output}.filt" ]]; then
	awk -F'\002' '{if($2!=0) print $0}' $model_output | sort -k2,2 -t$'\002' -r -g > ${model_output}.filt
fi

echo ""
echo "****************************** validation: model=$model_output, val_date=$data_day ******************************"
time java -Xmx70g -cp ftrl_maven_walter-0.0.1-SNAPSHOT.jar com.mobvista.ftrl.tools.OnlineReplay -data $data_path -conf conf/ -model ${model_output}.filt -out ./train_output/
echo "****************************** validation over: model=$model_output, val_date=$data_day *****************************"
echo ""
pwd

