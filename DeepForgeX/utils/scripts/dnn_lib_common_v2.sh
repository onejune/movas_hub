#!/bin/bash
set -e

################################################################################################
# dnn_lib_common_v2.sh - 重构版本
# 
# 改进点:
# 1. 不再复制 trainflows/*.py 和 tools/*.py 到实验目录
# 2. 通过 PYTHONPATH 直接引用 utils/ 目录
# 3. python.zip 只在不存在或过期时重新打包
# 4. 路径配置集中管理，支持环境变量覆盖
################################################################################################

# ============================================================
# 路径配置 (支持环境变量覆盖)
# ============================================================

# 自动检测 DeepForgeX 根目录
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DEEPFORGEX_ROOT="${DEEPFORGEX_ROOT:-$(dirname $(dirname $_SCRIPT_DIR))}"

# Python 环境
PYTHON_ENV="${PYTHON_ENV:-/root/anaconda3/envs/spore/bin/python}"
PYTHON_ENV_DIR="$(dirname $PYTHON_ENV)"

# MetaSpore 路径
METASPORE_DIR="${METASPORE_DIR:-$DEEPFORGEX_ROOT/MetaSpore/python}"
UTILS_DIR="${UTILS_DIR:-$DEEPFORGEX_ROOT/utils}"

# 本地目录
MODEL_OUTPUT_DIR="./output"
LOG_DIR="./log"
INTERRUPT_FILE="train_interrupt.flag"

# _metaspore.so 路径
METASPORE_SO_LOCAL="${METASPORE_DIR}/metaspore/_metaspore.so"
METASPORE_SO_FALLBACK="/mnt/data/oss_wanjun/03_online/_metaspore.so"

# 训练脚本路径 (由具体实验覆盖)
# 注意: 现在直接指向 utils/trainflows/ 而不是 ./src/
TRAINER_SCRIPT_PATH="${UTILS_DIR}/trainflows/dnn_trainFlow.py"

# ============================================================
# 初始化环境 (重构版)
# ============================================================
function init_env() {
    mkdir -p ${MODEL_OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p ./temp/
    
    local current_dir=$(pwd)
    CURRENT_PROJ_NAME=$(basename "$PWD")
    
    echo "=============================================="
    echo "DeepForgeX 训练环境初始化 (v2)"
    echo "=============================================="
    echo "DeepForgeX 根目录: $DEEPFORGEX_ROOT"
    echo "MetaSpore 目录: $METASPORE_DIR"
    echo "Utils 目录: $UTILS_DIR"
    echo "实验目录: $current_dir"
    echo "项目名称: $CURRENT_PROJ_NAME"
    echo "Python: $PYTHON_ENV"
    echo "=============================================="
    
    # 创建中断控制文件
    touch "$INTERRUPT_FILE"
    log "INFO" "中断控制文件已创建: $INTERRUPT_FILE"
    
    # ============================================================
    # 设置 PYTHONPATH (核心改动: 不再复制文件)
    # ============================================================
    # 优先级: utils/trainflows > utils/tools > MetaSpore > 原有 PYTHONPATH
    export PYTHONPATH="${UTILS_DIR}/trainflows:${UTILS_DIR}/tools:${METASPORE_DIR}:${PYTHONPATH}"
    export PATH="$PYTHON_ENV_DIR:$PATH"
    export PYSPARK_PYTHON=$PYTHON_ENV
    export PYSPARK_DRIVER_PYTHON=$PYTHON_ENV
    
    log "INFO" "PYTHONPATH 已设置: $PYTHONPATH"
    
    # ============================================================
    # 检查 _metaspore.so
    # ============================================================
    if [ ! -f "$METASPORE_SO_LOCAL" ]; then
        if [ -f "$METASPORE_SO_FALLBACK" ]; then
            cp "$METASPORE_SO_FALLBACK" "$METASPORE_SO_LOCAL"
            log "INFO" "_metaspore.so 拷贝成功"
        else
            log "ERROR" "_metaspore.so 不存在: $METASPORE_SO_FALLBACK"
            exit 1
        fi
    else
        log "INFO" "_metaspore.so 已存在，跳过"
    fi
    
    # ============================================================
    # python.zip 打包 (增量更新)
    # ============================================================
    _pack_python_zip "$current_dir"
    
    # 复制 kill_trainer.sh (这个保留，因为是运维脚本)
    cp "$UTILS_DIR/scripts/kill_trainer.sh" "$current_dir/" 2>/dev/null || true
    
    echo "环境初始化完成"
}

# ============================================================
# python.zip 智能打包 (只在需要时重新打包)
# ============================================================
function _pack_python_zip() {
    local target_dir="$1"
    local zip_file="$target_dir/python.zip"
    local marker_file="$target_dir/.python_zip_marker"
    
    # 检查是否需要重新打包
    local need_repack=false
    
    if [ ! -f "$zip_file" ]; then
        log "INFO" "python.zip 不存在，需要打包"
        need_repack=true
    elif [ ! -f "$marker_file" ]; then
        log "INFO" "标记文件不存在，需要重新打包"
        need_repack=true
    else
        # 检查 MetaSpore 和 utils 是否有更新
        local marker_time=$(stat -c %Y "$marker_file" 2>/dev/null || echo 0)
        local metaspore_time=$(find "$METASPORE_DIR" -name "*.py" -newer "$marker_file" 2>/dev/null | head -1)
        local utils_time=$(find "$UTILS_DIR" -name "*.py" -newer "$marker_file" 2>/dev/null | head -1)
        
        if [ -n "$metaspore_time" ] || [ -n "$utils_time" ]; then
            log "INFO" "检测到代码更新，需要重新打包"
            need_repack=true
        fi
    fi
    
    if [ "$need_repack" = true ]; then
        log "INFO" "打包 python.zip..."
        
        # 创建临时目录用于打包
        local tmp_pack_dir=$(mktemp -d)
        
        # 复制 MetaSpore python 目录
        cp -r "$METASPORE_DIR" "$tmp_pack_dir/python"
        
        # 复制 utils 到 python 目录下 (这样 Spark executor 也能访问)
        cp -r "$UTILS_DIR/trainflows" "$tmp_pack_dir/python/"
        cp -r "$UTILS_DIR/tools" "$tmp_pack_dir/python/"
        
        # 打包
        cd "$tmp_pack_dir"
        rm -f "$zip_file"
        zip -rq "$zip_file" python -x "*.pyc" -x "__pycache__/*" -x "*.so"
        cd "$target_dir"
        
        # 清理
        rm -rf "$tmp_pack_dir"
        
        # 更新标记文件
        touch "$marker_file"
        
        log "INFO" "python.zip 打包完成: $(du -h $zip_file | cut -f1)"
    else
        log "INFO" "python.zip 无需更新，跳过打包"
    fi
}

# ============================================================
# 日志函数
# ============================================================
function log() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "[${timestamp}] [${level}] ${message}"
}

# ============================================================
# 通用训练函数
# ============================================================
function model_train() { 
    init_env
    local conf_file="${1:-./conf/widedeep.yaml}"
    local eval_keys="${2:-business_type}"
    
    log "INFO" "启动训练: $TRAINER_SCRIPT_PATH"
    log "INFO" "配置文件: $conf_file"
    
    nohup env PYTHONUNBUFFERED=1 $PYTHON_ENV $TRAINER_SCRIPT_PATH \
        --name "$CURRENT_PROJ_NAME"  \
        --conf "$conf_file"  \
        --eval_keys "$eval_keys" \
        2>&1 | grep -v -E "bkdr_hash_combine|add expr|StringBKDRHash" | tee ${LOG_DIR}/log.log &
    
    echo "训练已启动, PID: $!"
    echo "日志: tail -f ${LOG_DIR}/log.log"
}

# ============================================================
# 验证函数
# ============================================================
function model_validation() {
    init_env
    local model_date="$1"
    local sample_date="$2"
    local eval_keys="${3:-business_type}"
    local conf_file="${4:-./conf/widedeep.yaml}"
    local shuffle_feature="${5:-}"
    
    local shuffle_arg=""
    if [ -n "$shuffle_feature" ]; then
        shuffle_arg="--shuffle_feature $shuffle_feature"
    fi
    
    log "INFO" "启动验证: model_date=$model_date, sample_date=$sample_date"
    
    nohup env PYTHONUNBUFFERED=1 $PYTHON_ENV $TRAINER_SCRIPT_PATH \
        --conf "$conf_file" \
        --validation True \
        --name "$CURRENT_PROJ_NAME" \
        --model_date "$model_date" \
        --sample_date "$sample_date" \
        --eval_keys "$eval_keys" \
        $shuffle_arg \
        2>&1 | grep -v -E "bkdr_hash_combine|add expr|StringBKDRHash" | tee ${LOG_DIR}/val.log &
    
    echo "验证已启动, PID: $!"
}

# ============================================================
# 环境检查
# ============================================================
function env_check() {
    echo "=========================================="
    echo "环境版本检查"
    echo "=========================================="
    echo "DeepForgeX: $DEEPFORGEX_ROOT"
    echo "MetaSpore: $METASPORE_DIR"
    echo "Utils: $UTILS_DIR"
    echo ""
    echo "Python 路径: $(which python)"
    echo "Python 版本: $($PYTHON_ENV --version 2>&1)"
    echo ""
    echo "PySpark 版本:"
    $PYTHON_ENV -c "import pyspark; print(pyspark.__version__)" 2>/dev/null || echo "未安装"
    echo ""
    echo "PyTorch 版本:"
    $PYTHON_ENV -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装"
    echo ""
    echo "MetaSpore:"
    $PYTHON_ENV -c "import metaspore; print(metaspore.__file__)" 2>/dev/null || echo "未安装"
    echo "=========================================="
}

# ============================================================
# 检查是否需要中断
# ============================================================
function should_interrupt() {
    [[ ! -f "$INTERRUPT_FILE" ]]
}
