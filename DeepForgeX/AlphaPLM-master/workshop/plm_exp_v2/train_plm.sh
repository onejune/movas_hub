para="-piece_num 6 -u_stdev 0.01 -u_alpha 0.1 -u_beta 1 -u_l1 1 -u_l2 100 -w_stdev 0.01 -w_alpha 0.1 -w_beta 1 -w_l1 1 -w_l2 100 -core 12"
para="-piece_num 6 -u_stdev 0.01 -u_alpha 0.05 -u_beta 1 -u_l1 1 -u_l2 100 -w_stdev 0.01 -w_alpha 0.05 -w_beta 1 -w_l1 1 -w_l2 100 -core 12"
para="-piece_num 6 -u_stdev 0.01 -u_alpha 0.05 -u_beta 1 -u_l1 1 -u_l2 100 -w_stdev 0.01 -w_alpha 0.05 -w_beta 1 -w_l1 1 -w_l2 100 -core 24"
model_out="./output/base"
train_dtm=$1
train_data="../../duf_outer/sample/sample_$1"

if [[ ! -f "$train_data" ]]; then
    echo "$train_data 不存在,训练终止!"
    exit 1
fi 

echo $para

if [[ ! -f "${model_out}" ]]; then
    cat ${train_data} | ../model_bin/plm_train $para -m ${model_out}
    #cat ${train_data} | awk -F'\002' '{if($13=="COM.ZZKKO") print $0}' | ../model_bin/plm_train $para -m ${model_out}
else
    cat ${train_data} | ../model_bin/plm_train $para -m ${model_out} -im ${model_out}
    #cat ${train_data} | awk -F'\002' '{if($13=="COM.ZZKKO") print $0}' | ../model_bin/plm_train $para -m ${model_out} -im ${model_out}
fi

cp ${model_out} ${model_out}.$train_dtm
echo "feature count: $(wc -l ${model_out})"

keep_days=7
cutoff_date=$(date -d "${train_dtm} - ${keep_days} days" +"%Y-%m-%d")
find "./output" -name "base.${cutoff_date}*" -delete
