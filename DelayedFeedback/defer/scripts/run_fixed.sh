#!/bin/bash
# 修复后的实验脚本
cd /mnt/workspace/walter.wan/open_research/defer

# 窗口配置: C=24h, win1=24h, win2=48h, win3=72h
COMMON_ARGS="--data_path ./data/business_64d_v3.txt \
    --batch_size 2048 --epoch 1 --lr 0.001 --l2_reg 1e-6 \
    --C 24 --win1 24 --win2 48 --win3 72 \
    --pre_train_start 0 --pre_train_end 30 \
    --pre_test_start 30 --pre_test_end 40 \
    --stream_start 30 --stream_mid 30 --stream_end 60 \
    --seed 42 --cache_path ./cache_pytorch/data.pkl"

mkdir -p logs_pytorch checkpoints_pytorch

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 修复后实验: 窗口 24h/48h/72h"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="

# ============ 预训练 (串行) ============
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始预训练..."

# Baseline 预训练
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 预训练: Baseline"
python3 ./src_pytorch/train.py --method Pretrain --mode pretrain \
    --save_path ./checkpoints_pytorch/baseline/model.pt \
    $COMMON_ARGS > logs_pytorch/pretrain_baseline.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Baseline 预训练完成!"

# ES-DFM 预训练 (使用相同的基础模型)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 预训练: ES-DFM"
python3 ./src_pytorch/train.py --method Pretrain --mode pretrain \
    --save_path ./checkpoints_pytorch/esdfm/model.pt \
    $COMMON_ARGS > logs_pytorch/pretrain_esdfm.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ES-DFM 预训练完成!"

# DFM 预训练
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 预训练: DFM"
python3 ./src_pytorch/train.py --method Pretrain --mode pretrain \
    --save_path ./checkpoints_pytorch/dfm/model.pt \
    $COMMON_ARGS > logs_pytorch/pretrain_dfm.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DFM 预训练完成!"

# WinAdapt 预训练
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 预训练: WinAdapt"
python3 ./src_pytorch/train.py --method Pretrain --mode pretrain \
    --save_path ./checkpoints_pytorch/winadapt/model.pt \
    $COMMON_ARGS > logs_pytorch/pretrain_winadapt.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WinAdapt 预训练完成!"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有预训练完成!"

# 更新预训练路径
PRETRAIN_ARGS="--pretrain_baseline_path ./checkpoints_pytorch/baseline/model.pt \
    --pretrain_esdfm_path ./checkpoints_pytorch/esdfm/model.pt \
    --pretrain_dfm_path ./checkpoints_pytorch/dfm/model.pt \
    --pretrain_winadapt_path ./checkpoints_pytorch/winadapt/model.pt"

# ============ 流式训练 (并行) ============
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始流式训练..."

# 第一批: Oracle, Vanilla (基线)
python3 ./src_pytorch/train.py --method Oracle --mode stream $COMMON_ARGS $PRETRAIN_ARGS > logs_pytorch/stream_Oracle.log 2>&1 &
python3 ./src_pytorch/train.py --method Vanilla --mode stream $COMMON_ARGS $PRETRAIN_ARGS > logs_pytorch/stream_Vanilla.log 2>&1 &
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Oracle, Vanilla 完成"

# 第二批: FNW, FNC
python3 ./src_pytorch/train.py --method FNW --mode stream $COMMON_ARGS $PRETRAIN_ARGS > logs_pytorch/stream_FNW.log 2>&1 &
python3 ./src_pytorch/train.py --method FNC --mode stream $COMMON_ARGS $PRETRAIN_ARGS > logs_pytorch/stream_FNC.log 2>&1 &
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] FNW, FNC 完成"

# 第三批: DFM, ES-DFM, WinAdapt
python3 ./src_pytorch/train.py --method DFM --mode stream $COMMON_ARGS $PRETRAIN_ARGS > logs_pytorch/stream_DFM.log 2>&1 &
python3 ./src_pytorch/train.py --method ES-DFM --mode stream $COMMON_ARGS $PRETRAIN_ARGS > logs_pytorch/stream_ES-DFM.log 2>&1 &
python3 ./src_pytorch/train.py --method delay_win_adapt --mode stream $COMMON_ARGS $PRETRAIN_ARGS > logs_pytorch/stream_winadapt.log 2>&1 &
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DFM, ES-DFM, WinAdapt 完成"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有流式训练完成!"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验结束: 窗口 24h/48h/72h"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="

# 汇总结果
echo ""
echo "=== 最终结果汇总 ==="
for model in Oracle Vanilla FNW FNC DFM ES-DFM winadapt; do
    log_file="logs_pytorch/stream_${model}.log"
    if [ -f "$log_file" ]; then
        echo "--- $model ---"
        grep "最终结果" -A 6 "$log_file" | tail -6
    fi
done
