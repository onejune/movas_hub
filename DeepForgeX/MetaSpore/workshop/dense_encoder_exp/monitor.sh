#!/bin/bash
# 监控 dense_exp 实验进度

EXP_DIR=/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/workshop/ctr/dense_exp

echo "=========================================="
echo "Dense 特征编码器对比实验"
echo "=========================================="
echo ""

for exp in baseline linear_noproject minmax standard log; do
    LOG=$EXP_DIR/$exp/nohup.log
    if [ -f "$LOG" ]; then
        # 检查是否完成
        if grep -q "Completed complete flow" "$LOG"; then
            # 提取 Overall AUC
            AUC=$(grep "Overall" "$LOG" | grep -oP "0\.\d+" | head -1)
            PCOC=$(grep "Overall" "$LOG" | grep -oP "0\.\d+" | head -2 | tail -1)
            echo "✅ $exp: 完成 (AUC=$AUC, PCOC=$PCOC)"
        else
            # 检查是否在训练
            if ps aux | grep "$exp" | grep -q "dnn_trainFlow"; then
                # 提取最新 AUC
                LAST_AUC=$(tail -100 "$LOG" | grep "auc:" | tail -1 | grep -oP "auc: 0\.\d+" | head -1)
                echo "🔄 $exp: 训练中... ($LAST_AUC)"
            else
                echo "⏳ $exp: 等待中"
            fi
        fi
    else
        echo "❓ $exp: 无日志"
    fi
done
