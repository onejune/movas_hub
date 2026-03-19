#!/bin/bash
source lib_common.sh

echo "[$(date '+%H:%M:%S')] === Java FTRL 7天训练 (02-11~02-17 train, 02-18 test) ==="
echo "[$(date '+%H:%M:%S')] 训练..."

# 7天训练
train_by_date 2026-02-11 2026-02-17

echo ""
echo "[$(date '+%H:%M:%S')] 验证 (02-18)..."
validation_ivr 2026-02-17 2026-02-18

echo ""
echo "[$(date '+%H:%M:%S')] === 7天训练完成 ==="
