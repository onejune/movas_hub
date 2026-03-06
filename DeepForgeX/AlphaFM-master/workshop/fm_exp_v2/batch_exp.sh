#!/bin/bash

# 模型参数配置
declare -A params=(
    ["param1"]="-init_stdev 0.1 -w_l2 10.0 -w_l1 1.0 -v_l1 0 -w_alpha 0.05 -v_l2 5.0 -v_alpha 0.05 -w_beta 1.0 -v_beta 1.0 -core 32 -dim 8,1,8"
    ["param2"]="-init_stdev 0.1 -w_l2 1.0 -w_l1 0.1 -v_l1 0 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.05 -w_beta 1.0 -v_beta 1.0 -core 16 -dim 8,1,8"
    ["param3"]="-init_stdev 0.001 -w_l2 5.0 -w_l1 1 -v_l1 0 -w_alpha 0.2 -v_l2 5.0 -v_alpha 0.2 -w_beta 1.0 -v_beta 1.0 -core 8 -dim 8,1,8"
    ["param4"]="-init_stdev 0.01 -w_l2 50.0 -w_l1 5 -v_l1 0.01 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.1 -w_beta 1.0 -v_beta 1.0 -core 8 -dim 8,1,8"
    ["param5"]="-init_stdev 0.01 -w_l2 50.0 -w_l1 5 -v_l1 0.01 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.1 -w_beta 1.0 -v_beta 1.0 -core 8 -dim 1,1,1"
    ["param6"]="-init_stdev 0.01 -w_l2 50.0 -w_l1 5 -v_l1 0.01 -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.1 -w_beta 1.0 -v_beta 1.0 -core 8 -dim 4,1,4"
)

# 设置起始日期和结束日期
start_date="20241201"
end_date="20250203"
val_date="20250204"

# 将日期转换为可以递增的格式
start_date=$(date -d "$start_date" +"%Y%m%d")
end_date=$(date -d "$end_date" +"%Y%m%d")
val_date=$(date -d "$val_date" +"%Y-%m-%d")

# 模型输出目录
model_output_dir="./output"

# 训练模型
train_model() {
    local train_date=$1
    local param_id=$2
    local para=$3

    local model_out="${model_output_dir}/base_${param_id}"
    local train_data="../../click/sample/sample_${train_date}"

    echo "=================== train date: $train_date, param: $param_id ===================="
    echo "Using parameters: $para"

    if [[ ! -f "$train_data" ]]; then
        echo "$train_data 不存在,训练终止!"
        exit 1
    fi

    if [[ ! -f "${model_out}" ]]; then
        cat ${train_data} | ../model_bin/fm_train $para -m ${model_out}
    else
        cat ${train_data} | ../model_bin/fm_train $para -m ${model_out} -im ${model_out}
    fi

    # 备份模型
    cp ${model_out} ${model_out}.${train_date}
}

# 验证模型
validate_model() {
    local val_date=$1
    local param_id=$2

    local model_out="${model_output_dir}/base_${param_id}"
    local train_data="../../click/sample/sample_${val_date}"
    local para=${params[$param_id]}
    
    echo "=================== validation date: $val_date, param: $param_id ==================="
    cat ${train_data} | awk -F'\002' '{if($13=="COM.ZZKKO") print $0}' | ../model_bin/fm_predict -m ${model_out} -out ${model_out}.predict -core 8

    if [ -f "${model_out}.predict" ]; then
        cat ${model_out}.predict | python figure_auc.py
    fi
}

# 清理旧模型
cleanup_old_models() {
    local train_date=$1
    local param_id=$2
    local keep_days=7

    local cutoff_date=$(date -d "${train_date} - ${keep_days} days" +"%Y-%m-%d")
    find "${model_output_dir}" -name "base_${param_id}.${cutoff_date}*" -delete
}

# 主函数
main() {
    # 遍历每一组参数
    for param_id in "${!params[@]}"; do
        local para=${params[$param_id]}
        local current_date=$start_date

        # 训练阶段
        while [ "$current_date" -le "$end_date" ]; do
            train_date=$(date -d "$current_date" +"%Y-%m-%d")
            time train_model $train_date $param_id "$para"
            cleanup_old_models $train_date $param_id
            current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
        done

        # 验证阶段
        time validate_model $val_date $param_id
    done
}

# 执行主函数
main
