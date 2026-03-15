#!/bin/bash
set -e

# 配置区域
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
MODEL_OUTPUT_DIR="./output"
LOG_DIR="./log"
OSSUTIL_CONFIG="./ossutilconfig"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d%H%M%S).log"
################################################################################################
# 系统默认的metaspore库在 /root/anaconda3/envs/spore/lib/python3.8/site-packages/metaspore/
# 这是通过 pip install metaspore 安装的
# 如果自己开发模型代码，需要先 pip uninstall metaspore
# 然后在MetaSpore目录下执行 pip install -e . 
# 会在系统的site-packages下生成一个软链接: metaspore.egg-info 指向真实的 ms 路径，或者直接修改 egg-info指向自己的开发路径
# 需要保证真实的 ms 路径下有"metaspore/algos/"目录
################################################################################################
METASPORE_DIR="/mnt/workspace/walter.wan/dnn/MetaSpore/python/" # 模型依赖的 metaspore.algos在这个路径下
TRAINER_SCRIPT_PATH="./dnn_trainFlow.py"

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

    export PYTHONPATH=$METASPORE_DIR

    #获取当前执行目录
    current_dir=$(pwd)
    echo "current_dir: ${current_dir}"
    # 获取dnn_lib_common.sh 脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    echo "dnn_lib_common.sh 所在的目录是: $SCRIPT_DIR"
    CURRENT_PROJ_NAME=$(basename "$PWD")
    echo "项目名称: $CURRENT_PROJ_NAME"

    cp $SCRIPT_DIR/movas_logger.py $METASPORE_DIR/metaspore/
    cp $SCRIPT_DIR/dnn_trainFlow.py $current_dir/
    cp $SCRIPT_DIR/winrate_trainFlow.py $current_dir/
    cp $SCRIPT_DIR/ltv_trainFlow.py $current_dir/
    cp $SCRIPT_DIR/MDL_trainFlow.py $current_dir/
    cp $SCRIPT_DIR/MTL_trainFlow.py $current_dir/
    cp $SCRIPT_DIR/metrics_eval.py $current_dir/
    cp $SCRIPT_DIR/movas_logger.py $current_dir/
    cp $SCRIPT_DIR/feishu_notifier.py $current_dir/
    cp $SCRIPT_DIR/python.zip $current_dir/
    cp $SCRIPT_DIR/*.sh $current_dir/
    echo "开始后台运行......"
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
    init_env
    nohup env PYTHONUNBUFFERED=1 $PYTHON_ENV $TRAINER_SCRIPT_PATH \
        --name "$CURRENT_PROJ_NAME"  \
        --conf ./conf/widedeep.yaml  \
        --eval_keys "business_type,is_ifa_null,objective_type" \
        2>&1 | grep -v -E "bkdr_hash_combine|add expr|StringBKDRHash" > nohup.log &
}

function model_validation() {
    init_env
    keys="$3"
    if [ -z "$3" ]; then
        keys="business_type,is_ifa_null,objective_type"
    fi
    $PYTHON_ENV $TRAINER_SCRIPT_PATH \
        --conf ./conf/widedeep.yaml \
        --validation True \
        --name "$CURRENT_PROJ_NAME" \
        --model_date "$1" \
        --sample_date "$2" \
        --eval_keys $keys
}