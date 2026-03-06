# File 2: run_train.sh (主执行脚本)
#!/bin/bash

source ./lib_common.sh

current_date=$1
local_data_path="../sample/sample_${current_date}"
MODEL_OUTPUT_DIR="./train_output"
ftrl_jar_path="/mnt/workspace/walter.wan/ftrl_in_git/target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar"

if [[ ! -f "$local_data_path" ]]; then
    log "ERROR" "数据准备失败，跳过训练"
fi

# 执行训练
log "INFO" "启动模型训练..."
time java -jar -Xmx80g $ftrl_jar_path \
    -i "${local_data_path}" \
    -c "./conf" \
    -f "f" \
    -n "${current_date}"

if [ $? -ne 0 ]; then
    log "ERROR" "模型训练失败"
    exit 2
fi

# 后处理
log "INFO" "最终结果过滤..."
cp "${MODEL_OUTPUT_DIR}/base" "${MODEL_OUTPUT_DIR}/base.${current_date}"
