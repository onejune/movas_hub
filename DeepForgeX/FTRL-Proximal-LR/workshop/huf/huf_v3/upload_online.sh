model_oss="oss://spark-ml-train-new/liufashuai/dsp/cps/purchase_model/8_lr_model"
model_dtm=$1

output_model="online_model_$model_dtm"
input_model="train_output/base.$model_dtm"

echo -e "fea_key\002fea_w">$output_model
awk -F'\002' '{if($2!=0) print $1"\002"$2}' $input_model>>$output_model

line_num=$(wc -l < $output_model)
echo "line_num for filted model: $line_num"

if [[ $line_num -le 10000 ]]; then
    echo "invalid model."
    exit 1
fi

echo "model transfer successfully, begin upload to $model_oss"
ossutil cp -f $input_model $model_oss/model_file_$model_dtm
ossutil cp -f $output_model $model_oss/$output_model

mv $output_model ./train_output/
