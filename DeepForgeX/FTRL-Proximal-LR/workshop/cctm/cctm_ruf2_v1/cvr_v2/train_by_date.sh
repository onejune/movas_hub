# File 2: run_train.sh (主执行脚本)
#!/bin/bash

source ./lib_common.sh

# 参数校验
if [ $# -ne 2 ]; then
    echo "用法: $0 <开始日期> <结束日期>"
    echo "示例: $0 2024-08-30 2025-02-04"
    exit 1
fi

start_date=$(format_date "$1")
end_date=$(format_date "$2")
current_date=${start_date}
sample_path="/mnt/data/oss_sample/ruf_sample_v1/sample_ruf_v2"

# 初始化环境
init_env
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d%H%M%S).log"

# 主训练循环
while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
    log "INFO" "======== 开始训练 ${current_date} ========"
    
    # 准备训练数据
    local_data_path=$(find $sample_path/$current_date -maxdepth 1 -type f -name "*.csv")
    if [[ ! -f "$local_data_path" ]]; then
        log "ERROR" "数据准备失败，跳过本日训练"
        current_date=$(date -d "${current_date} + 1 day" +"%Y-%m-%d")
        continue
    fi

    # 执行训练
    log "INFO" "启动模型训练..."
    time java -jar -Xmx80g ../../ftrl_maven_walter-0.0.1-SNAPSHOT.jar \
        -i "${local_data_path}" \
        -c "./conf" \
        -f "f" \
	-n "$current_date"
    
    if [ $? -ne 0 ]; then
        log "ERROR" "模型训练失败"
        exit 2
    fi

    old_date=$(date -d "${current_date} - 14 day" +"%Y-%m-%d")
    if [[ -f "${MODEL_OUTPUT_DIR}/base.${old_date}" ]]; then
	log "INFO" "filt old feature......"
	python filter_old_feature.py ${MODEL_OUTPUT_DIR}/base.${old_date} ${MODEL_OUTPUT_DIR}/base > train_output/filted_feature.dat
	mv "${MODEL_OUTPUT_DIR}/base.tmp" "${MODEL_OUTPUT_DIR}/base"
    fi
    # 保存模型输出
    cp "${MODEL_OUTPUT_DIR}/base" "${MODEL_OUTPUT_DIR}/base.${current_date}"
    log "INFO" "模型保存完成: ${MODEL_OUTPUT_DIR}/base.${current_date}"

    # 清理旧模型
    clean_old_models

    current_date=$(date -d "${current_date} + 1 day" +"%Y-%m-%d")
done

# 后处理
log "INFO" "最终结果过滤..."
awk -F'\002' '$2!=0' "${MODEL_OUTPUT_DIR}/base" | sort -t$'\002' -k2,2 -rg > "${MODEL_OUTPUT_DIR}/base.filt"
log "INFO" "训练完成，有效记录数: $(wc -l < ${MODEL_OUTPUT_DIR}/base.filt)"
