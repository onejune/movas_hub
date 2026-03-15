#!/bin/bash
cd /mnt/workspace/walter.wan/open_research/defer

COMMON_ARGS="--data_path ./data/business_64d_v3.txt \
    --batch_size 2048 --epoch 1 --lr 0.001 --l2_reg 1e-6 \
    --C 6 --win1 6 --win2 24 --win3 72 \
    --pre_train_start 0 --pre_train_end 30 \
    --pre_test_start 30 --pre_test_end 40 \
    --stream_start 30 --stream_mid 30 --stream_end 60 \
    --seed 42 --cache_path ./cache_pytorch/data.pkl \
    --pretrain_baseline_path ./checkpoints_pytorch/pretrain_baseline.pt \
    --pretrain_esdfm_path ./checkpoints_pytorch/pretrain_esdfm.pt \
    --pretrain_dfm_path ./checkpoints_pytorch/pretrain_dfm.pt \
    --pretrain_winadapt_path ./checkpoints_pytorch/pretrain_winadapt.pt"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始补充训练: DFM, ES-DFM, WinAdapt"

# DFM
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 流式训练: DFM"
python3 ./src_pytorch/train.py --method DFM --mode stream $COMMON_ARGS > logs_pytorch/stream_DFM_v2.log 2>&1 &
DFM_PID=$!

# ES-DFM
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 流式训练: ES-DFM"
python3 ./src_pytorch/train.py --method ES-DFM --mode stream $COMMON_ARGS > logs_pytorch/stream_ES-DFM_v2.log 2>&1 &
ESDFM_PID=$!

# WinAdapt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 流式训练: WinAdapt"
python3 ./src_pytorch/train.py --method delay_win_adapt --mode stream $COMMON_ARGS > logs_pytorch/stream_winadapt.log 2>&1 &
WINADAPT_PID=$!

wait $DFM_PID
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DFM 完成"

wait $ESDFM_PID  
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ES-DFM 完成"

wait $WINADAPT_PID
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WinAdapt 完成"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有补充训练完成!"
