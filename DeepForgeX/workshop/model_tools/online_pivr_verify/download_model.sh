project=ruf2_v5
model_dtm=$1

model_dir="/mnt/data/oss_dsp_algo/ivr/model/$project"
model_path="$model_path/online_model_$model_dtm"

cp -r $model_dir/conf ./ 
cp $model_path ./data/
