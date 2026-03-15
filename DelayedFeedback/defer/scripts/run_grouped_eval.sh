#!/bin/bash
# 分组评估实验 - 比较不同延迟窗口下各方法的效果

set -e
cd /mnt/workspace/walter.wan/open_research/defer

LOG_DIR="./logs_pytorch"
mkdir -p $LOG_DIR

echo "=========================================="
echo "分组评估实验 - $(date)"
echo "=========================================="

# 方法列表
METHODS=("Vanilla" "Oracle" "FNC" "delay_win_adapt")

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo ">>> 训练 $METHOD ..."
    
    if [ "$METHOD" == "delay_win_adapt" ]; then
        PRETRAIN_ARG="--pretrain_winadapt_path ./checkpoints_pytorch/winadapt/model.pt"
    else
        PRETRAIN_ARG="--pretrain_baseline_path ./checkpoints_pytorch/baseline/model.pt"
    fi
    
    python3 ./src_pytorch/train.py --method $METHOD --mode stream \
        --data_path ./data/business_64d_v3.txt \
        --batch_size 2048 --epoch 1 --lr 0.001 --l2_reg 1e-6 \
        --C 24 --win1 24 --win2 48 --win3 72 \
        --stream_start 30 --stream_mid 30 --stream_end 60 \
        --seed 42 \
        $PRETRAIN_ARG \
        2>&1 | tee $LOG_DIR/grouped_${METHOD}.log
    
    echo ">>> $METHOD 完成"
done

echo ""
echo "=========================================="
echo "所有实验完成"
echo "=========================================="

# 汇总结果
echo ""
echo ">>> 结果汇总:"
for METHOD in "${METHODS[@]}"; do
    echo ""
    echo "=== $METHOD ==="
    grep -A 10 "最终结果" $LOG_DIR/grouped_${METHOD}.log | head -15
done
