#!/bin/bash
cd /mnt/workspace/walter.wan/open_research/defer

# 新窗口配置: C=24h, win1=24h, win2=48h, win3=72h
COMMON_ARGS="--data_path ./data/business_64d_v3.txt \
    --batch_size 2048 --epoch 1 --lr 0.001 --l2_reg 1e-6 \
    --C 24 --win1 24 --win2 48 --win3 72 \
    --pre_train_start 0 --pre_train_end 30 \
    --pre_test_start 30 --pre_test_end 40 \
    --stream_start 30 --stream_mid 30 --stream_end 60 \
    --seed 42 --cache_path ./cache_pytorch/data.pkl"

mkdir -p logs_pytorch checkpoints_pytorch

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 新实验: 窗口 24h/48h/72h"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="

# ============ 预训练 (串行) ============
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始预训练..."

for model in Baseline ES-DFM WinAdapt DFM; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 预训练: $model"
    python3 ./src_pytorch/train.py --method Pretrain --mode pretrain \
        --save_path ./checkpoints_pytorch/pretrain_$(echo $model | tr '[:upper:]' '[:lower:]' | tr '-' '_').pt \
        $COMMON_ARGS > logs_pytorch/pretrain_$(echo $model | tr '[:upper:]' '[:lower:]' | tr '-' '_').log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model 预训练完成!"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有预训练完成!"

# 更新预训练路径
PRETRAIN_ARGS="--pretrain_baseline_path ./checkpoints_pytorch/pretrain_baseline.pt \
    --pretrain_esdfm_path ./checkpoints_pytorch/pretrain_es_dfm.pt \
    --pretrain_dfm_path ./checkpoints_pytorch/pretrain_dfm.pt \
    --pretrain_winadapt_path ./checkpoints_pytorch/pretrain_winadapt.pt"

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
