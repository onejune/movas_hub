#!/bin/bash
set -e

# 配置区域
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
PYTHON_ENV_DIR="/root/anaconda3/envs/spore/bin"

MODEL_OUTPUT_DIR="./output"
LOG_DIR="./log"
OSSUTIL_CONFIG="./ossutilconfig"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d%H%M%S).log"

################################################################################################
# MetaSpore 路径配置
# 使用 DeepForgeX 内的 MetaSpore (包含 DEFER 等自定义模型)
################################################################################################
METASPORE_DIR="/mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/MetaSpore/python"

# _metaspore.so OSS 路径（超过 100M，不纳入 git，启动时自动下载）
METASPORE_SO_OSS="oss://spark-ml-train-new/wanjun/03_online/_metaspore.so"
METASPORE_SO_LOCAL="${METASPORE_DIR}/metaspore/_metaspore.so"

# 训练脚本路径 (由具体实验覆盖)
TRAINER_SCRIPT_PATH="./src/dnn_trainFlow.py"

# 定义中断控制文件路径
INTERRUPT_FILE="train_interrupt.flag"

# 初始化环境
function init_env() {
    mkdir -p ${MODEL_OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p ./temp/
    mkdir -p ./src/
    LOGFILE="${LOG_DIR}/script_$(date +%Y%m%d).log"
    echo "python_env: ${PYTHON_ENV}"
    echo "log_file: ${LOG_FILE}"
    touch "$INTERRUPT_FILE"
    log "INFO" "中断控制文件已创建: $INTERRUPT_FILE"

    export PYTHONPATH=$METASPORE_DIR:$PYTHONPATH
    export PATH="$PYTHON_ENV_DIR:$PATH"
    export PYSPARK_PYTHON=$PYTHON_ENV
    export PYSPARK_DRIVER_PYTHON=$PYTHON_ENV

    # 检查并下载 _metaspore.so（不纳入 git，从 OSS 自动拉取）
    if [ ! -f "$METASPORE_SO_LOCAL" ]; then
        log "INFO" "_metaspore.so 不存在，尝试从 OSS 下载: $METASPORE_SO_OSS"
        ossutil cp "$METASPORE_SO_OSS" "$METASPORE_SO_LOCAL" 2>/dev/null
        if [ $? -eq 0 ]; then
            log "INFO" "_metaspore.so 下载成功 (OSS)"
        else
            log "WARN" "OSS 下载失败，尝试从本地挂载路径拷贝: /mnt/data/oss_wanjun/03_online/_metaspore.so"
            cp "/mnt/data/oss_wanjun/03_online/_metaspore.so" "$METASPORE_SO_LOCAL" 2>/dev/null
            if [ $? -eq 0 ]; then
                log "INFO" "_metaspore.so 拷贝成功 (本地挂载)"
            else
                log "ERROR" "_metaspore.so 获取失败，已尝试以下路径："
                log "ERROR" "  1. OSS: $METASPORE_SO_OSS"
                log "ERROR" "  2. 本地: /mnt/data/oss_wanjun/03_online/_metaspore.so"
                log "ERROR" "请检查 OSS 权限或确认本地挂载路径是否正确"
                exit 1
            fi
        fi
    else
        log "INFO" "_metaspore.so 已存在，跳过下载"
    fi

    # 获取当前执行目录
    current_dir=$(pwd)
    echo "current_dir: ${current_dir}"
    
    # 获取 dnn_lib_common.sh 脚本所在目录 (DeepForgeX/utils/scripts)
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    UTILS_DIR="$(dirname "$SCRIPT_DIR")"
    echo "utils 目录: $UTILS_DIR"
    
    CURRENT_PROJ_NAME=$(basename "$PWD")
    echo "项目名称: $CURRENT_PROJ_NAME"

    # 复制依赖到实验目录 (从新的子目录结构)
    cp $UTILS_DIR/tools/movas_logger.py $METASPORE_DIR/metaspore/
    cp $UTILS_DIR/trainflows/*.py $current_dir/src/
    cp $UTILS_DIR/tools/*.py $current_dir/src/
    cp $UTILS_DIR/scripts/kill_trainer.sh $current_dir/ 2>/dev/null || true
    
    # 打包 python.zip (Spark executor 需要)
    echo "打包 python.zip..."
    cd $METASPORE_DIR/..
    rm -f python.zip
    zip -rq python.zip python -x "*.pyc" -x "__pycache__/*"
    mv python.zip $current_dir/
    cd $current_dir
    
    echo "环境初始化完成"
}

function env_check() {
    echo "=========================================="
    echo "环境版本检查"
    echo "=========================================="

    echo "Python 路径: $(which python)"
    echo "Python 版本:"
    python3 --version 2>/dev/null || python --version 2>/dev/null || echo "未找到 Python"
    echo ""

    echo "PySpark 版本:"
    python3 -c "import pyspark; print(pyspark.__version__)" 2>/dev/null || echo "未安装 PySpark"
    echo ""

    echo "Torch 版本:"
    python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装 Torch"
    echo ""

    echo "MetaSpore 路径: $METASPORE_DIR"
    echo "=========================================="
}

# 检查是否需要中断
function should_interrupt() {
    [[ ! -f "$INTERRUPT_FILE" ]]
}

# 日志记录函数
function log() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "[${timestamp}] [${level}] ${message}"
}

# 日期格式转换
function format_date() {
    local input_date=$1
    date -d "${input_date}" +"%Y-%m-%d"
}

# 通用训练函数
function model_train() { 
    init_env
    local conf_file="${1:-./conf/widedeep.yaml}"
    local eval_keys="${2:-business_type}"
    
    nohup env PYTHONUNBUFFERED=1 $PYTHON_ENV $TRAINER_SCRIPT_PATH \
        --name "$CURRENT_PROJ_NAME"  \
        --conf "$conf_file"  \
        --eval_keys "$eval_keys" \
        2>&1 | grep -v -E "bkdr_hash_combine|add expr|StringBKDRHash" | tee ${LOG_DIR}/log.log &
    
    echo "训练已启动, PID: $!"
    echo "日志: tail -f ${LOG_DIR}/train.log"
}

# 验证函数
function model_validation() {
    init_env
    local model_date="$1"
    local sample_date="$2"
    local eval_keys="${3:-business_type}"
    local conf_file="${4:-./conf/config.yaml}"
    
    nohup env PYTHONUNBUFFERED=1 $PYTHON_ENV $TRAINER_SCRIPT_PATH \
        --conf "$conf_file" \
        --validation True \
        --name "$CURRENT_PROJ_NAME" \
        --model_date "$model_date" \
        --sample_date "$sample_date" \
        --eval_keys "$eval_keys" \
        2>&1 | grep -v -E "bkdr_hash_combine|add expr|StringBKDRHash" | tee ${LOG_DIR}/val.log &
}
