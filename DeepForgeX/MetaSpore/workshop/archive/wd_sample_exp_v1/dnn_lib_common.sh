#!/bin/bash
set -e

# 配置区域
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
MODEL_OUTPUT_DIR="./output"
LOG_DIR="./log"
OSSUTIL_CONFIG="./ossutilconfig"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d%H%M%S).log"
METASPORE_DIR="../MetaSpore"

# 定义中断控制文件路径
INTERRUPT_FILE="train_interrupt.flag"

# 初始化环境
function init_env() {
    mkdir -p ${MODEL_OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p ./temp/
    LOGFILE="${LOG_DIR}/script_$(date +%Y%m%d).log"
    echo "python_env: ${PYTHON_ENV}"
    echo "log_file: ${LOG_FILE}"
    echo "oss_sample_path: ${OSS_SAMPLE_PATH}"
    echo "ftrl_jar_path: ${ftrl_jar_path}"
    touch "$INTERRUPT_FILE"
    log "INFO" "中断控制文件已创建: $INTERRUPT_FILE"

    #获取当前执行目录
    current_dir=$(pwd)
    echo "current_dir: ${current_dir}"
    # 获取dnn_lib_common.sh 脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    echo "dnn_lib_common.sh 所在的目录是: $SCRIPT_DIR"
    CURRENT_PROJ_NAME=$(basename "$PWD")
    echo "项目名称: $CURRENT_PROJ_NAME"

    cp $SCRIPT_DIR/ms_dnn_wd.py $current_dir/
    cp $SCRIPT_DIR/metrics_eval.py $current_dir/
    cp $SCRIPT_DIR/movas_logger.py $current_dir/
    cp $SCRIPT_DIR/feishu_notifier.py $current_dir/
    cp $SCRIPT_DIR/python.zip $current_dir/

    # 检查MetaSpore目录是否存在
    if [ ! -d "$METASPORE_DIR" ]; then
        echo "MetaSpore目录不存在，正在从 $SCRIPT_DIR/ms.tar.gz 解压..."
        # 检查压缩包是否存在
        if [ ! -f "$SCRIPT_DIR/ms.tar.gz" ]; then
            echo "错误: $SCRIPT_DIR/ms.tar.gz 文件不存在"
            exit 1
        fi
        if tar -xzf "$SCRIPT_DIR/ms.tar.gz" -C "$current_dir/.."; then
            echo "解压成功"
        else
            echo "解压失败"
            exit 1
        fi
    else
        echo "MetaSpore目录已存在: $METASPORE_DIR"
    fi
}

function env_check() {
    echo "=========================================="
    echo "环境版本检查"
    echo "=========================================="

    # 检查 Python 版本
    echo "python 路径：$(which python)"
    echo "Python 版本:"
    if command -v python3 &> /dev/null; then
        python3 --version
    elif command -v python &> /dev/null; then
        python --version
    else
        echo "未找到 Python"
    fi
    echo ""

    # 检查 PySpark 版本
    echo "PySpark 版本:"
    if python3 -c "import pyspark; print(pyspark.__version__)" 2> /dev/null; then
        :
    elif python -c "import pyspark; print(pyspark.__version__)" 2> /dev/null; then
        :
    else
        echo "未安装 PySpark 或无法导入"
    fi
    echo ""

    # 检查 Torch 版本
    echo "Torch 版本:"
    if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2> /dev/null; then
        :
    elif python -c "import torch; print('PyTorch version:', torch.__version__)" 2> /dev/null; then
        :
    else
        echo "未安装 Torch 或无法导入"
    fi
    echo ""

    # 检查 Conda 环境
    echo "Conda 环境信息:"
    if command -v conda &> /dev/null; then
        echo "Conda 版本:"
        conda --version
        echo ""
        echo "当前 Conda 环境:"
        conda info --envs
        echo ""
        echo "激活的 Conda 环境:"
        if [ -n "$CONDA_DEFAULT_ENV" ]; then
            echo "$CONDA_DEFAULT_ENV"
        else
            echo "base"
        fi
    else
        echo "未找到 Conda"
    fi
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

function model_train() { 
    #init_env
    $PYTHON_ENV ms_dnn_wd.py --conf ./conf/widedeep.yaml --name "$CURRENT_PROJ_NAME" --eval_keys "business_type,is_ifa_null,objective_type"
}

function model_validation() {
    init_env
    keys="$3"
    if [ -z "$3" ]; then
        keys="business_type,is_ifa_null,objective_type"
    fi
    $PYTHON_ENV ms_dnn_wd.py --conf ./conf/widedeep.yaml --validation True --name "$CURRENT_PROJ_NAME" --model "$1" --sample "$2" --eval_keys $keys
}
