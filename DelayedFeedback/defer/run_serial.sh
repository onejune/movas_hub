#!/bin/bash
# 串行训练脚本 - 一次只跑一个模型避免 OOM

cd /mnt/workspace/walter.wan/open_research/defer
LOG_DIR="logs_tf_v2"

echo "=========================================="
echo "串行训练开始: $(date)"
echo "=========================================="

# 1. Vanilla
echo ""
echo "[1/3] 开始训练 Vanilla..."
python3 src_tf_v2/train.py --method vanilla --batch_size 4096 > ${LOG_DIR}/vanilla.log 2>&1
VANILLA_EXIT=$?
echo "Vanilla 完成，退出码: $VANILLA_EXIT ($(date))"

# 2. Oracle  
echo ""
echo "[2/3] 开始训练 Oracle..."
python3 src_tf_v2/train.py --method oracle --batch_size 4096 > ${LOG_DIR}/oracle.log 2>&1
ORACLE_EXIT=$?
echo "Oracle 完成，退出码: $ORACLE_EXIT ($(date))"

# 3. WinAdapt
echo ""
echo "[3/3] 开始训练 WinAdapt..."
python3 src_tf_v2/train.py --method winadapt --batch_size 4096 > ${LOG_DIR}/winadapt.log 2>&1
WINADAPT_EXIT=$?
echo "WinAdapt 完成，退出码: $WINADAPT_EXIT ($(date))"

echo ""
echo "=========================================="
echo "全部训练完成: $(date)"
echo "退出码: Vanilla=$VANILLA_EXIT, Oracle=$ORACLE_EXIT, WinAdapt=$WINADAPT_EXIT"
echo "=========================================="
