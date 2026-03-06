#!/bin/bash

source ./lib_common.sh

sample_path=$1
local_data_path="${sample_path}"
MODEL_OUTPUT_DIR="./train_output"

if [[ ! -f "$local_data_path" ]]; then
    log "ERROR" "数据准备失败，跳过训练"
fi

# 执行训练
log "INFO" "启动模型训练..."
time java -jar -Xmx80g ../../ftrl_maven_walter-0.0.1-SNAPSHOT.jar \
    -i "${local_data_path}" \
    -c "./conf" \
    -f "f" \
    -n "${current_date}"

if [ $? -ne 0 ]; then
    log "ERROR" "模型训练失败"
    exit 2
fi

