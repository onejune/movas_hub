#!/bin/bash
# 用业务数据运行 Defer 实验（适配短时间跨度）

set -e

# 配置
DATA_PATH="${DATA_PATH:-./data/business_64d_v3.txt}"
CACHE_PATH="./cache_business"
CKPT_PATH="./checkpoints_business"
LOG_PATH="./logs_business"
SRC_DIR="./src"

BATCH_SIZE="${BATCH_SIZE:-1024}"
EPOCH="${EPOCH:-3}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"
C="${C:-6}"
WIN1="${WIN1:-6}"
WIN2="${WIN2:-24}"
WIN3="${WIN3:-72}"
RNWIN="${RNWIN:-24}"

# 时间配置（70天数据：前35天预训练，后35天流式）
PRE_START=0
PRE_MID=35
PRE_END=70
START=0
MID=35
END=70

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

setup_dirs() {
    mkdir -p "$CACHE_PATH" "$CKPT_PATH" "$LOG_PATH"
    mkdir -p "$CKPT_PATH/baseline" "$CKPT_PATH/esdfm" "$CKPT_PATH/winadapt" "$CKPT_PATH/wintime"
    log "目录创建完成"
}

check_data() {
    if [ ! -f "$DATA_PATH" ]; then
        echo "错误: 数据文件不存在: $DATA_PATH"
        echo "请先运行: python convert_data.py --input_dir <数据目录> --output $DATA_PATH"
        exit 1
    fi
    log "数据文件检查通过: $DATA_PATH"
}

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
        --pre_start "$PRE_START" \
        --pre_mid "$PRE_MID" \
        --pre_end "$PRE_END" \
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
        --pre_start "$PRE_START" \
        --pre_mid "$PRE_MID" \
        --pre_end "$PRE_END" \
        2>&1 | tee "$LOG_PATH/pretrain_esdfm.log"
    log "ES-DFM 预训练完成"
}

pretrain_winadapt() {
    log "========== 开始预训练 WinAdapt 模型 =========="
    python3 "$SRC_DIR/main.py" \
        --method delay_win_adapt \
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
        --pre_start "$PRE_START" \
        --pre_mid "$PRE_MID" \
        --pre_end "$PRE_END" \
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
        --pre_start "$PRE_START" \
        --pre_mid "$PRE_MID" \
        --pre_end "$PRE_END" \
        2>&1 | tee "$LOG_PATH/pretrain_wintime.log"
    log "WinTime 预训练完成"
}

stream_vanilla() {
    log "========== 开始流式训练 Vanilla =========="
    python3 "$SRC_DIR/main.py" \
        --method Vanilla \
        --mode stream \
        --data_path "$DATA_PATH" \
        --data_cache_path "$CACHE_PATH" \
        --pretrain_baseline_model_ckpt_path "$CKPT_PATH/baseline" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        --start "$START" \
        --mid "$MID" \
        --end "$END" \
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
        --start "$START" \
        --mid "$MID" \
        --end "$END" \
        2>&1 | tee "$LOG_PATH/stream_fnw.log"
    log "FNW 流式训练完成"
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
        --start "$START" \
        --mid "$MID" \
        --end "$END" \
        2>&1 | tee "$LOG_PATH/stream_esdfm.log"
    log "ES-DFM 流式训练完成"
}

stream_delay_win_adapt() {
    log "========== 开始流式训练 Delay Win Adapt (Defer) =========="
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
        --start "$START" \
        --mid "$MID" \
        --end "$END" \
        2>&1 | tee "$LOG_PATH/stream_delay_win_adapt.log"
    log "Delay Win Adapt 流式训练完成"
}

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
    stream_vanilla
    stream_fnw
    stream_esdfm
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

usage() {
    cat << EOF
业务数据 Defer 实验脚本

用法: $0 <command>

命令:
  all                 运行完整流程 (预训练 + 流式训练)
  pretrain            运行所有预训练
  stream              运行所有流式训练
  
  pretrain_baseline   预训练 Baseline 模型
  pretrain_esdfm      预训练 ES-DFM 模型
  pretrain_winadapt   预训练 WinAdapt 模型
  pretrain_wintime    预训练 WinTime 模型
  
  stream_vanilla      流式训练 Vanilla
  stream_fnw          流式训练 FNW
  stream_esdfm        流式训练 ES-DFM
  stream_delay_win_adapt  流式训练 Delay Win Adapt (Defer)

环境变量:
  DATA_PATH           数据文件路径 (默认: ./data/business_data.txt)
  BATCH_SIZE          批大小 (默认: 1024)
  EPOCH               预训练轮数 (默认: 3)
  LR                  学习率 (默认: 0.001)

时间配置 (7天数据):
  预训练: 第 0-4 天
  流式训练: 第 4-7 天

示例:
  # 运行完整流程
  ./run_business.sh all
  
  # 仅预训练
  ./run_business.sh pretrain
  
  # 仅运行 Defer 方法
  ./run_business.sh stream_delay_win_adapt
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
    stream_vanilla)
        check_data
        stream_vanilla
        ;;
    stream_fnw)
        check_data
        stream_fnw
        ;;
    stream_esdfm)
        check_data
        stream_esdfm
        ;;
    stream_delay_win_adapt)
        check_data
        stream_delay_win_adapt
        ;;
    *)
        usage
        exit 1
        ;;
esac
