#!/bin/bash
set -e

# 配置区域
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
PYTHON_ENV_DIR="/root/anaconda3/envs/spore/bin"

MODEL_OUTPUT_DIR="./output"
LOG_DIR="./log"
OSSUTIL_CONFIG="./ossutilconfig"

METASPORE_DIR="/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/python"
# _metaspore.so OSS 路径（超过 100M，不纳入 git，启动时自动下载）
METASPORE_SO_OSS="oss://spark-ml-train-new/wanjun/03_online/_metaspore.so"
METASPORE_SO_LOCAL="${METASPORE_DIR}/metaspore/_metaspore.so"

# 定义中断控制文件路径
INTERRUPT_FILE="train_interrupt.flag"

# 初始化环境
function init_env() {
    mkdir -p ${MODEL_OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
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
    else
        log "INFO" "_metaspore.so 已存在，跳过下载"
    fi

    # 获取当前执行目录
    current_dir=$(pwd)
    echo "current_dir: ${current_dir}"
    
    # 获取项目名称
    CURRENT_PROJ_NAME=$(basename "$PWD")
    echo "项目名称: $CURRENT_PROJ_NAME"
    CURRENT_MODEL_TYPE=$(basename "$(dirname "$PWD")")
    echo "模型类型: $CURRENT_MODEL_TYPE"

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

# 使用包结构的训练函数
function model_train() { 
    init_env
    local conf_file="${1:-./conf/widedeep.yaml}"
    local eval_keys="${2:-business_type}"
    
    nohup env PYTHONUNBUFFERED=1 $PYTHON_ENV ./train.py \
        --name "$CURRENT_PROJ_NAME"  \
        --conf "$conf_file"  \
        --eval_keys "$eval_keys" \
        2>&1 | grep -v -E "bkdr_hash_combine|add expr|StringBKDRHash" | tee ${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log &
    
    echo "训练已启动, PID: $!"
    echo "日志: tail -f ${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
}

# 使用包结构的验证函数
function model_validation() {
    init_env
    local model_date="$1"
    local sample_date="$2"
    local conf_file="${3:-./conf/config.yaml}"
    local eval_keys="${4:-business_type}"
    local shuffle_feature="${5:-}"
    
    local shuffle_arg=""
    if [ -n "$shuffle_feature" ]; then
        shuffle_arg="--shuffle_feature $shuffle_feature"
    fi
    
    nohup env PYTHONUNBUFFERED=1 $PYTHON_ENV ./train.py \
        --conf "$conf_file" \
        --validation True \
        --name "$CURRENT_PROJ_NAME" \
        --model_date "$model_date" \
        --sample_date "$sample_date" \
        --eval_keys "$eval_keys" \
        $shuffle_arg \
        2>&1 | grep -v -E "bkdr_hash_combine|add expr|StringBKDRHash" | tee ${LOG_DIR}/validation_$(date +%Y%m%d_%H%M%S).log &
}

# 显示帮助信息
function show_help() {
    echo "Usage: $0 [train|validate|help] [options...]"
    echo "  train     : 训练模型 (默认)"
    echo "  validate  : 验证模型"
    echo "  help      : 显示此帮助信息"
    echo ""
    echo "训练参数: $0 train [conf_file] [eval_keys]"
    echo "验证参数: $0 validate [model_date] [sample_date] [conf_file] [eval_keys] [shuffle_feature]"
}

# 主逻辑
case "${1:-train}" in
    train)
        shift
        model_train "$@"
        ;;
    validate)
        shift
        model_validation "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        # 默认为训练模式
        model_train "$@"
        ;;
esac