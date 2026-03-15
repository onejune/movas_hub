#!/bin/bash
# TF 原版全量实验脚本
cd /mnt/workspace/walter.wan/open_research/defer
export TF_CPP_MIN_LOG_LEVEL=3

CACHE_DIR="./cache_tf"
CKPT_DIR="./checkpoints_tf"
LOG_DIR="./logs_tf"

mkdir -p "$CACHE_DIR" "$LOG_DIR"
mkdir -p "$CKPT_DIR/baseline" "$CKPT_DIR/esdfm" "$CKPT_DIR/dfm" "$CKPT_DIR/winadapt" "$CKPT_DIR/wintime"

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }

run_tf() {
    local method="$1" mode="$2" extra_args="$3" log="$4"
    echo "$(ts) >>> $method ($mode)"
    (cd src_tf && python3 main.py \
        --method "$method" --mode "$mode" \
        --data_path "../data/business_64d_v3.txt" \
        --data_cache_path "../cache_tf" \
        --batch_size 2048 --epoch 1 --lr 0.001 --l2_reg 1e-6 \
        --C 24 --win1 24 --win2 48 --win3 72 --seed 42 \
        --pretrain_baseline_model_ckpt_path "../checkpoints_tf/baseline" \
        --pretrain_esdfm_model_ckpt_path "../checkpoints_tf/esdfm" \
        --pretrain_dfm_model_ckpt_path "../checkpoints_tf/dfm" \
        --pretrain_winselect_model_ckpt_path "../checkpoints_tf/winadapt" \
        --pretrain_wintime_model_ckpt_path "../checkpoints_tf/wintime" \
        $extra_args) > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "$(ts) !!! $method 失败 (rc=$rc)，查看 $log"
        tail -5 "$log" | grep -v "oneDNN\|AVX\|rebuild\|cuda\|I0000\|E0000\|WARNING"
    else
        echo "$(ts) <<< $method 完成"
    fi
    return $rc
}

echo "$(ts) =========================================="
echo "$(ts) TF 原版全量实验"
echo "$(ts) =========================================="

# ============ 预训练 ============
echo "$(ts) --- 预训练阶段 ---"

# Baseline 已完成，跳过
if [ -f "$CKPT_DIR/baseline/weights.weights.h5" ]; then
    echo "$(ts) Baseline ckpt 已存在，跳过预训练"
else
    run_tf Pretrain pretrain "--model_ckpt_path ../checkpoints_tf/baseline" "$LOG_DIR/pretrain_baseline.log"
fi

run_tf ES-DFM    pretrain "--model_ckpt_path ../checkpoints_tf/esdfm"    "$LOG_DIR/pretrain_esdfm.log"    || true
run_tf DFM       pretrain "--model_ckpt_path ../checkpoints_tf/dfm"      "$LOG_DIR/pretrain_dfm.log"      || true
run_tf win_adapt pretrain "--model_ckpt_path ../checkpoints_tf/winadapt" "$LOG_DIR/pretrain_winadapt.log" || true

echo "$(ts) 所有预训练完成"

# ============ 流式训练 ============
echo "$(ts) --- 流式训练阶段 ---"

run_tf Oracle          stream "" "$LOG_DIR/stream_Oracle.log"   || true
run_tf FNC             stream "" "$LOG_DIR/stream_FNC.log"      || true
run_tf Vanilla         stream "" "$LOG_DIR/stream_Vanilla.log"  || true
run_tf ES-DFM          stream "" "$LOG_DIR/stream_ESDFM.log"    || true
run_tf DFM             stream "" "$LOG_DIR/stream_DFM.log"      || true
run_tf delay_win_adapt stream "" "$LOG_DIR/stream_WinAdapt.log" || true

echo "$(ts) =========================================="
echo "$(ts) 所有实验完成！"
echo "$(ts) =========================================="

# ============ 汇总 ============
echo ""
echo "=== TF 原版实验结果汇总 ==="
for method in Oracle FNC Vanilla ESDFM DFM WinAdapt; do
    log="$LOG_DIR/stream_${method}.log"
    [ -f "$log" ] || continue
    echo "--- $method ---"
    python3 -c "
with open('$log','rb') as f:
    t = f.read().decode('utf-8',errors='replace')
lines = [l.strip() for l in t.split('\n') if 'auc' in l.lower() and 'test' in l.lower()]
for l in lines[-3:]: print(l)
" 2>/dev/null
done
