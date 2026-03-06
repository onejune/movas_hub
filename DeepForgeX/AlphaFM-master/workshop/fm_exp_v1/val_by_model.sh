
model_date=$(date -d "$1" +"%Y-%m-%d")
sample_date=$(date -d "$2" +"%Y-%m-%d")

model_path="./output/base.$model_date"
train_data="../../huf_rafactor/sample_offline/sample_$sample_date"

if [[ ! -f "$train_data" ]]; then
    echo "$train_data 不存在,训练终止!"
fi 

cat ${train_data} | awk -F'\002' '{if($13=="COM.ZZKKO") print $0}' | ../model_bin/fm_predict $para -m ${model_path} -out ${model_path}.predict -core 8 -dim 8


echo "=================== validation result: model=${model_path} data=$2 ======================"
if [ -f "${model_path}.predict" ]; then
    cat ${model_path}.predict | python figure_auc.py 
fi

