#!/bin/bash
# ============================================================================
# 一键运行全流程: Step3 → Step4 → Step5
# （Step1和Step2需要先手动执行一次）
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "一键运行全流程"
echo "=========================================="

# Step 3: 文本 → Embedding
echo ""
echo ">>> 执行 Step 3: 文本 → Embedding"
bash "${SCRIPT_DIR}/step3_text2emb.sh"

# Step 4: 训练 RQ-VAE
echo ""
echo ">>> 执行 Step 4: 训练 RQ-VAE"
bash "${SCRIPT_DIR}/step4_train_rqvae.sh"

echo ""
echo "=========================================="
echo "⚠️  Step 4 完成后，请手动操作:"
echo "  1. 查看 output/checkpoints/ 下的最佳模型路径"
echo "  2. 将路径填入 config.sh 的 RQVAE_CKPT_PATH"
echo "  3. 然后执行: bash step5_generate_sid.sh"
echo "=========================================="
