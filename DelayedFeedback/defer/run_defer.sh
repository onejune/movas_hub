#!/bin/bash
# =============================================================================
# Defer 完整复现脚本
# Real Negatives Matter: Continuous Training with Real Negatives 
# for Delayed Feedback Modeling (KDD 2021)
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置区域 - 根据实际情况修改
# =============================================================================

# 数据路径 (Criteo 数据集)
DATA_PATH="${DATA_PATH:-/mnt/workspace/walter.wan/open_research/defer/data/criteo_data.txt}"

# 缓存和模型保存路径
CACHE_PATH="${CACHE_PATH:-/mnt/workspace/walter.wan/open_research/defer/cache}"
CKPT_PATH="${CKPT_PATH:-/mnt/workspace/walter.wan/open_research/defer/checkpoints}"
LOG_PATH="${LOG_PATH:-/mnt/workspace/walter.wan/open_research/defer/logs}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-1024}"
EPOCH="${EPOCH:-5}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"

# 时间窗口参数
C="${C:-0.25}"           # 等待窗口 (小时)
WIN1="${WIN1:-0.25}"     # 窗口1 (小时)
WIN2="${WIN2:-0.5}"      # 窗口2 (小时)
WIN3="${WIN3:-1}"        # 窗口3 (小时)
RNWIN="${RNWIN:-24}"     # 真负样本确认窗口 (小时)

# 代码目录
SRC_DIR="$(dirname "$0")/src"

# =============================================================================
# 辅助函数
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_data() {
    if [[ ! -f "$DATA_PATH" ]]; then
        echo "=============================================="
        echo "错误: 未找到数据文件: $DATA_PATH"
        echo ""
        echo "请下载 Criteo 延迟反馈数据集:"
        echo "  https://labs.criteo.com/2013/12/conversion-logs-dataset/"
        echo ""
        echo "下载后设置 DATA_PATH 环境变量:"
        echo "  export DATA_PATH=/path/to/criteo/data.txt"
        echo "=============================================="
        exit 1
    fi
    log "数据文件检查通过: $DATA_PATH"
}

setup_dirs() {
    mkdir -p "$CACHE_PATH"
    mkdir -p "$CKPT_PATH"/{baseline,esdfm,winadapt,wintime,fsiw0,fsiw1,dfm}
    mkdir -p "$LOG_PATH"
    log "目录结构创建完成"
}

# =============================================================================
# 预训练阶段
# =============================================================================

pretrain_baseline() {
    log "========== 开始预训练 Baseline 模型 =========="
    python3 "$SRC_DIR/main.py" \
        --method Pretrain \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --model_ckpt_path "$CKPT_PATH/baseline" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/pretrain_baseline.log"
    log "Baseline 预训练完成"
}

pretrain_esdfm() {
    log "========== 开始预训练 ES-DFM 模型 =========="
    python3 "$SRC_DIR/main.py" \
        --method ES-DFM \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --model_ckpt_path "$CKPT_PATH/esdfm" \
        --C "$C" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/pretrain_esdfm.log"
    log "ES-DFM 预训练完成"
}

pretrain_winadapt() {
    log "========== 开始预训练 WinAdapt 模型 =========="
    python3 "$SRC_DIR/main.py" \
        --method win_adapt \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --model_ckpt_path "$CKPT_PATH/winadapt" \
        --C "$C" \
        --win1 "$WIN1" \
        --win2 "$WIN2" \
        --win3 "$WIN3" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/pretrain_winadapt.log"
    log "WinAdapt 预训练完成"
}

pretrain_wintime() {
    log "========== 开始预训练 WinTime 模型 =========="
    python3 "$SRC_DIR/main.py" \
        --method win_time \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --model_ckpt_path "$CKPT_PATH/wintime" \
        --C "$C" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/pretrain_wintime.log"
    log "WinTime 预训练完成"
}

pretrain_dfm() {
    log "========== 开始预训练 DFM 模型 =========="
    python3 "$SRC_DIR/main.py" \
        --method DFM \
        --mode pretrain \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --model_ckpt_path "$CKPT_PATH/dfm" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/pretrain_dfm.log"
    log "DFM 预训练完成"
}

# =============================================================================
# 流式训练阶段
# =============================================================================

stream_dfm() {
    log "========== 开始流式训练 DFM =========="
    python3 "$SRC_DIR/main.py" \
        --method DFM \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_dfm_model_ckpt_path "$CKPT_PATH/dfm" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_dfm.log"
    log "DFM 流式训练完成"
}

stream_oracle() {
    log "========== 开始流式训练 Oracle (上界) =========="
    python3 "$SRC_DIR/main.py" \
        --method Oracle \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_oracle.log"
    log "Oracle 流式训练完成"
}

stream_vanilla() {
    log "========== 开始流式训练 Vanilla =========="
    python3 "$SRC_DIR/main.py" \
        --method Vanilla \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --C "$C" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_vanilla.log"
    log "Vanilla 流式训练完成"
}

stream_fnw() {
    log "========== 开始流式训练 FNW =========="
    python3 "$SRC_DIR/main.py" \
        --method FNW \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_fnw.log"
    log "FNW 流式训练完成"
}

stream_fnc() {
    log "========== 开始流式训练 FNC =========="
    python3 "$SRC_DIR/main.py" \
        --method FNC \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_fnc.log"
    log "FNC 流式训练完成"
}

stream_esdfm() {
    log "========== 开始流式训练 ES-DFM =========="
    python3 "$SRC_DIR/main.py" \
        --method ES-DFM \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --pretrain_esdfm_model_ckpt_path "$CKPT_PATH/esdfm" \
        --C "$C" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_esdfm.log"
    log "ES-DFM 流式训练完成"
}

stream_delay_win_time() {
    log "========== 开始流式训练 Delay Win Time =========="
    python3 "$SRC_DIR/main.py" \
        --method delay_win_time \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --pretrain_wintime_model_ckpt_path "$CKPT_PATH/wintime" \
        --C "$C" \
        --rnwin "$RNWIN" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_delay_win_time.log"
    log "Delay Win Time 流式训练完成"
}

stream_delay_win_adapt() {
    log "========== 开始流式训练 Delay Win Adapt (Defer 核心方法) =========="
    python3 "$SRC_DIR/main.py" \
        --method delay_win_adapt \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --pretrain_winselect_model_ckpt_path "$CKPT_PATH/winadapt" \
        --pretrain_wintime_model_ckpt_path "$CKPT_PATH/wintime" \
        --C "$C" \
        --win1 "$WIN1" \
        --win2 "$WIN2" \
        --win3 "$WIN3" \
        --rnwin "$RNWIN" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_PATH/stream_delay_win_adapt.log"
    log "Delay Win Adapt 流式训练完成"
}

# =============================================================================
# 主流程
# =============================================================================

run_all_pretrain() {
    log "========== 开始所有预训练 =========="
    pretrain_baseline
    pretrain_esdfm
    pretrain_winadapt
    pretrain_wintime
    log "========== 所有预训练完成 =========="
}

run_all_stream() {
    log "========== 开始所有流式训练 =========="
    stream_oracle
    stream_vanilla
    stream_fnw
    stream_fnc
    stream_esdfm
    stream_delay_win_time
    stream_delay_win_adapt
    log "========== 所有流式训练完成 =========="
}

run_all() {
    check_data
    setup_dirs
    run_all_pretrain
    run_all_stream
    log "========== 全部实验完成 =========="
}

# =============================================================================
# 命令行入口
# =============================================================================

usage() {
    cat << EOF
Defer 复现脚本

用法: $0 <command>

命令:
  all                 运行完整流程 (预训练 + 流式训练)
  pretrain            运行所有预训练
  stream              运行所有流式训练
  
  pretrain_baseline   预训练 Baseline 模型
  pretrain_esdfm      预训练 ES-DFM 模型
  pretrain_winadapt   预训练 WinAdapt 模型
  pretrain_wintime    预训练 WinTime 模型
  pretrain_dfm        预训练 DFM 模型
  
  stream_oracle       流式训练 Oracle (上界)
  stream_vanilla      流式训练 Vanilla
  stream_fnw          流式训练 FNW
  stream_fnc          流式训练 FNC
  stream_esdfm        流式训练 ES-DFM
  stream_delay_win_time   流式训练 Delay Win Time
  stream_delay_win_adapt  流式训练 Delay Win Adapt (Defer)
  
  setup               仅创建目录结构
  check               检查数据文件

环境变量:
  DATA_PATH           数据文件路径 (默认: $DATA_PATH)
  CACHE_PATH          缓存目录 (默认: $CACHE_PATH)
  CKPT_PATH           模型保存目录 (默认: $CKPT_PATH)
  BATCH_SIZE          批大小 (默认: $BATCH_SIZE)
  EPOCH               预训练轮数 (默认: $EPOCH)
  LR                  学习率 (默认: $LR)

示例:
  # 运行完整流程
  DATA_PATH=/path/to/data.txt ./run_defer.sh all
  
  # 仅预训练
  ./run_defer.sh pretrain
  
  # 仅运行 Defer 核心方法
  ./run_defer.sh pretrain_baseline
  ./run_defer.sh pretrain_winadapt
  ./run_defer.sh stream_delay_win_adapt
EOF
}

case "${1:-}" in
    all)
        run_all
        ;;
    pretrain)
        check_data
        setup_dirs
        run_all_pretrain
        ;;
    stream)
        check_data
        run_all_stream
        ;;
    pretrain_baseline)
        check_data
        setup_dirs
        pretrain_baseline
        ;;
    pretrain_esdfm)
        check_data
        setup_dirs
        pretrain_esdfm
        ;;
    pretrain_winadapt)
        check_data
        setup_dirs
        pretrain_winadapt
        ;;
    pretrain_wintime)
        check_data
        setup_dirs
        pretrain_wintime
        ;;
    pretrain_dfm)
        check_data
        setup_dirs
        pretrain_dfm
        ;;
    stream_oracle)
        check_data
        stream_oracle
        ;;
    stream_vanilla)
        check_data
        stream_vanilla
        ;;
    stream_fnw)
        check_data
        stream_fnw
        ;;
    stream_fnc)
        check_data
        stream_fnc
        ;;
    stream_esdfm)
        check_data
        stream_esdfm
        ;;
    stream_delay_win_time)
        check_data
        stream_delay_win_time
        ;;
    stream_delay_win_adapt)
        check_data
        stream_delay_win_adapt
        ;;
    stream_dfm)
        check_data
        stream_dfm
        ;;
    setup)
        setup_dirs
        ;;
    check)
        check_data
        ;;
    *)
        usage
        exit 1
        ;;
esac
