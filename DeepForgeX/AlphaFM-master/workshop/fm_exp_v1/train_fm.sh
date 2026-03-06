
para="-init_stdev 0.1 -w_l2 10.0 -w_l1 1.0 -v_l1 0 -w_alpha 0.05 -v_l2 5.0 -v_alpha 0.05 -w_beta 1.0 -v_beta 1.0 -core 32 -dim 1,1,8"
para="-init_stdev 0.1 -w_l2 1.0 -w_l1 0.1 -v_l1 0 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.05 -w_beta 1.0 -v_beta 1.0 -core 16 -dim 1,1,8"
para="-init_stdev 0.001 -w_l2 5.0 -w_l1 1 -v_l1 0 -w_alpha 0.2 -v_l2 5.0 -v_alpha 0.2 -w_beta 1.0 -v_beta 1.0 -core 8 -dim 1,1,8"
para="-init_stdev 0.01 -w_l2 50.0 -w_l1 5 -v_l1 0.01 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.1 -w_beta 1.0 -v_beta 1.0 -core 8 -dim 1,1,8"
para="-init_stdev 0.01 -w_l2 30.0 -w_l1 0.1 -v_l1 0.01 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.1 -w_beta 1.0 -v_beta 1.0 -core 8 -dim 1,1,1"
para="-init_stdev 0.01 -w_l2 5.0 -w_l1 5 -v_l1 0.1 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.01 -w_beta 1.0 -v_beta 1.0 -core 16 -dim 1,1,8"

model_out="./output/base"
train_dtm=$1
train_data="../../huf_rafactor/sample_offline/sample_$1"

if [[ ! -f "$train_data" ]]; then
    echo "$train_data 不存在,训练终止!"
    exit 1
fi 

echo $para

if [[ ! -f "${model_out}" ]]; then
    cat ${train_data} | ../model_bin/fm_train $para -m ${model_out}
    #cat ${train_data} | awk -F'\002' '{if($13=="COM.ZZKKO") print $0}' | ../model_bin/fm_train $para -m ${model_out}
else
    cat ${train_data} | ../model_bin/fm_train $para -m ${model_out} -im ${model_out}
    #cat ${train_data} | awk -F'\002' '{if($13=="COM.ZZKKO") print $0}' | ../model_bin/fm_train $para -m ${model_out} -im ${model_out}
fi

cp ${model_out} ${model_out}.$train_dtm
echo "feature count: $(wc -l ${model_out})"

keep_days=7
cutoff_date=$(date -d "${train_dtm} - ${keep_days} days" +"%Y-%m-%d")
find "./output" -name "base.${cutoff_date}*" -delete
