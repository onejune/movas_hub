model_output="./train_output/base.$1"
data_path=$1

filt_file="${model_output}.filt"

# 检查 filt 文件是否存在且不为空
if [[ ! -f "$filt_file" ]] || [[ ! -s "$filt_file" ]]; then
    # 如果 filt 文件不存在或为空，则执行 awk 和 sort 命令
    awk -F'\002' '{if($2!=0) print $0}' "$model_output" | sort -k2,2 -t$'\002' -r -g > "$filt_file"
else
    # 可选：如果需要处理 filt 文件存在且不为空的情况，可以在这里添加逻辑
    echo "File $filt_file exists and is not empty. Skipping processing."
fi

echo ""
echo "********************** validation begin: model=$model_output, val_date=$data_day ******************************"
time java -Xmx70g -cp ../../ftrl_maven_walter-0.0.1-SNAPSHOT.jar com.mobvista.ftrl.tools.OnlineReplay -data $data_path -conf conf/ -model ${model_output}.filt -out ./train_output/ -name "val-$data_day"
echo "********************** validation end: model=$model_output, val_date=$data_day *****************************"
echo ""
pwd

