#!/bin/bash
# PyTorch 并行训练脚本
# 同时运行多个方法，充分利用 32 核 CPU

set -e

# ============================================================================
# 配置
# ============================================================================

DATA_PATH="${DATA_PATH:-./data/business_64d_v3.txt}"
CACHE_PATH="./cache_pytorch"
CKPT_PATH="./checkpoints_pytorch"
LOG_PATH="./logs_pytorch"
SRC_DIR="./src_pytorch"

BATCH_SIZE="${BATCH_SIZE:-2048}"  # 增大 batch size
EPOCH="${EPOCH:-1}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"
C="${C:-6}"                       # 观察窗口 (小时)
WIN1="${WIN1:-6}"                 # 第一档窗口 (小时)
WIN2="${WIN2:-24}"                # 第二档窗口 (小时)
WIN3="${WIN3:-72}"                # 第三档窗口 (小时)
NUM_WORKERS="${NUM_WORKERS:-4}"   # DataLoader workers

# 时间配置
# 预训练：0-30天训练，30-40天测试
# 流式训练：30-60天
PRE_TRAIN_START=0
PRE_TRAIN_END=30
PRE_TEST_START=30
PRE_TEST_END=40
STREAM_START=30
STREAM_END=60

# 并行度控制
MAX_PARALLEL="${MAX_PARALLEL:-2}"  # 最多同时运行的任务数（降低避免 OOM）

# ============================================================================
# 辅助函数
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

setup_dirs() {
    mkdir -p "$CACHE_PATH" "$CKPT_PATH" "$LOG_PATH"
    mkdir -p "$CKPT_PATH/baseline" "$CKPT_PATH/esdfm" "$CKPT_PATH/winadapt" "$CKPT_PATH/dfm"
    log "目录创建完成"
}

check_data() {
    if [ ! -f "$DATA_PATH" ]; then
        echo "错误: 数据文件不存在: $DATA_PATH"
        exit 1
    fi
    log "数据文件检查通过: $DATA_PATH ($(du -h "$DATA_PATH" | cut -f1))"
}

wait_for_jobs() {
    # 等待后台任务数降到指定值以下
    local max_jobs=$1
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 5
    done
}

# ============================================================================
# 预训练函数
# ============================================================================

pretrain_baseline() {
    log "========== 开始预训练 Baseline 模型 =========="
    python3 "$SRC_DIR/train.py" \
        --method Pretrain \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --cache_path "$CACHE_PATH/data.pkl" \
        --model_ckpt_path "$CKPT_PATH/baseline" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        --pre_train_start "$PRE_TRAIN_START" \
        --pre_train_end "$PRE_TRAIN_END" \
        --pre_test_start "$PRE_TEST_START" --pre_test_end "$PRE_TEST_END" \
        2>&1 | tee "$LOG_PATH/pretrain_baseline.log"
    log "Baseline 预训练完成"
}

pretrain_esdfm() {
    log "========== 开始预训练 ES-DFM 模型 =========="
    python3 "$SRC_DIR/train.py" \
        --method ES-DFM \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --cache_path "$CACHE_PATH/data.pkl" \
        --model_ckpt_path "$CKPT_PATH/esdfm" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        --C "$C" \
        --pre_train_start "$PRE_TRAIN_START" \
        --pre_train_end "$PRE_TRAIN_END" \
        --pre_test_start "$PRE_TEST_START" --pre_test_end "$PRE_TEST_END" \
        2>&1 | tee "$LOG_PATH/pretrain_esdfm.log"
    log "ES-DFM 预训练完成"
}

pretrain_winadapt() {
    log "========== 开始预训练 WinAdapt 模型 =========="
    python3 "$SRC_DIR/train.py" \
        --method delay_win_adapt \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --cache_path "$CACHE_PATH/data.pkl" \
        --model_ckpt_path "$CKPT_PATH/winadapt" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        --C "$C" \
        --win1 "$WIN1" \
        --win2 "$WIN2" \
        --win3 "$WIN3" \
        --pre_train_start "$PRE_TRAIN_START" \
        --pre_train_end "$PRE_TRAIN_END" \
        --pre_test_start "$PRE_TEST_START" --pre_test_end "$PRE_TEST_END" \
        2>&1 | tee "$LOG_PATH/pretrain_winadapt.log"
    log "WinAdapt 预训练完成"
}

pretrain_dfm() {
    log "========== 开始预训练 DFM 模型 =========="
    python3 "$SRC_DIR/train.py" \
        --method DFM \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --cache_path "$CACHE_PATH/data.pkl" \
        --model_ckpt_path "$CKPT_PATH/dfm" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        --C "$C" \
        --pre_train_start "$PRE_TRAIN_START" \
        --pre_train_end "$PRE_TRAIN_END" \
        --pre_test_start "$PRE_TEST_START" --pre_test_end "$PRE_TEST_END" \
        2>&1 | tee "$LOG_PATH/pretrain_dfm.log"
    log "DFM 预训练完成"
}

# ============================================================================
# 流式训练函数
# ============================================================================

stream_train() {
    local method=$1
    local pretrain_path=$2
    local log_file="$LOG_PATH/stream_${method}.log"
    
    log "========== 开始流式训练: $method =========="
    
    python3 "$SRC_DIR/train.py" \
        --method "$method" \
        --mode stream \
        --data_path "$DATA_PATH" \
        --cache_path "$CACHE_PATH/data.pkl" \
        --pretrain_baseline_path "$CKPT_PATH/baseline/model.pt" \
        --pretrain_esdfm_path "$CKPT_PATH/esdfm/model.pt" \
        --pretrain_dfm_path "$CKPT_PATH/dfm/model.pt" \
        --pretrain_winadapt_path "$CKPT_PATH/winadapt/model.pt" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        --C "$C" \
        --win1 "$WIN1" \
        --win2 "$WIN2" \
        --win3 "$WIN3" \
        --stream_start "$STREAM_START" \
        --stream_mid "$STREAM_START" \
        --stream_end "$STREAM_END" \
        2>&1 | tee "$log_file"
    
    log "流式训练完成: $method"
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    log "============================================================"
    log "PyTorch 并行训练开始"
    log "数据: $DATA_PATH"
    log "并行度: $MAX_PARALLEL"
    log "Batch Size: $BATCH_SIZE"
    log "============================================================"
    
    setup_dirs
    check_data
    
    # ========== 阶段 1: 串行预训练（避免评估时 OOM）==========
    log ""
    log "========== 阶段 1: 串行预训练 =========="
    
    # 串行执行，评估时内存安全
    pretrain_baseline
    log "Baseline 预训练完成!"
    
    pretrain_esdfm
    log "ES-DFM 预训练完成!"
    
    pretrain_winadapt
    log "WinAdapt 预训练完成!"
    
    pretrain_dfm
    log "DFM 预训练完成!"
    
    log "所有预训练完成!"
    
    # ========== 阶段 2: 并行流式训练 ==========
    log ""
    log "========== 阶段 2: 并行流式训练 =========="
    
    # 定义所有流式训练方法
    METHODS=("Oracle" "Vanilla" "FNW" "FNC" "DFM" "ES-DFM" "delay_win_adapt")
    
    # 并行启动所有方法
    for method in "${METHODS[@]}"; do
        wait_for_jobs $MAX_PARALLEL
        stream_train "$method" &
        log "已启动: $method (PID: $!)"
        sleep 2  # 错开启动时间，避免同时读取数据
    done
    
    log "等待所有流式训练完成..."
    wait
    log "所有流式训练完成!"
    
    # ========== 汇总结果 ==========
    log ""
    log "============================================================"
    log "训练完成! 结果汇总:"
    log "============================================================"
    
    for method in "${METHODS[@]}"; do
        log_file="$LOG_PATH/stream_${method}.log"
        if [ -f "$log_file" ]; then
            echo ""
            echo "=== $method ==="
            grep -E "(AUC|PR-AUC|LogLoss|ECE|最终结果)" "$log_file" | tail -5
        fi
    done
    
    log ""
    log "详细日志: $LOG_PATH/"
}

# ============================================================================
# 单独运行选项
# ============================================================================

case "${1:-all}" in
    pretrain)
        setup_dirs
        check_data
        pretrain_baseline &
        pretrain_esdfm &
        pretrain_winadapt &
        pretrain_dfm &
        wait
        ;;
    stream)
        setup_dirs
        check_data
        METHODS=("Oracle" "Vanilla" "FNW" "FNC" "DFM" "ES-DFM" "delay_win_adapt")
        for method in "${METHODS[@]}"; do
            wait_for_jobs $MAX_PARALLEL
            stream_train "$method" &
            sleep 2
        done
        wait
        ;;
    single)
        # 单独运行某个方法: ./run_pytorch_parallel.sh single ES-DFM
        setup_dirs
        check_data
        stream_train "${2:-Vanilla}"
        ;;
    all|*)
        main
        ;;
esac

log "脚本执行完毕"
