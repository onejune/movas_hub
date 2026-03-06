# File 1: lib_common.sh (公共函数库)
#!/bin/bash

# 配置区域
OSS_ROOT="oss://spark-ml-train-new/liufashuai/dsp/cps/purchase_model/7_train_data"
MODEL_OUTPUT_DIR="./train_output"
LOG_DIR="./log"
OSSUTIL_CONFIG="./ossutilconfig"

# 初始化环境
init_env() {
    mkdir -p ${MODEL_OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p "./temp"
}

# 日志记录函数
log() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] [${level}] ${message}"
}

# 日期格式转换
format_date() {
    local input_date=$1
    date -d "${input_date}" +"%Y-%m-%d"
}

# 下载训练数据
download_data() {
    local data_day=$1
    local local_path=$2
    
    if [[ -f "${local_path}" ]]; then
        log "INFO" "本地数据已存在: ${local_path}"
        return 0
    fi

    log "INFO" "开始下载数据: ${data_day}"
    ossutil64 cp "${OSS_ROOT}/${data_day}/" "./temp/" --recursive --config-file ${OSSUTIL_CONFIG} || return 1
    
    find "./temp" -type f -name "*.csv" -exec cat {} >> ${local_path} \;
    if [ $? -ne 0 ]; then
        log "ERROR" "数据合并失败"
        return 2
    fi
    
    log "INFO" "数据下载完成: ${local_path}"
    return 0
}

# 清理旧模型
clean_old_models() {
    local keep_days=7
    local cutoff_date=$(date -d "${current_date} - ${keep_days} days" +"%Y-%m-%d")
    find ${MODEL_OUTPUT_DIR} -name "base.${cutoff_date}*" -delete
    log "INFO" "已清理 ${cutoff_date} 之前的模型"
}
