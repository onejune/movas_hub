#!/bin/bash
# ============================================================================
# Step 4: 训练 RQ-VAE 模型
# 输入:  embedding .npy 文件
# 输出:  RQ-VAE checkpoint (.pth)
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "Step 4: 训练 RQ-VAE"
echo "=========================================="
echo "  设备:     ${DEVICE_TRAIN_RQVAE}"
echo "  输入:     ${EMB_NPY_PATH}"
echo "  输出目录: ${OUTPUT_CHECKPOINTS}"
echo "  Epochs:   ${TRAIN_EPOCHS}"
echo "=========================================="

# 检查输入文件
if [ ! -f "${EMB_NPY_PATH}" ]; then
    echo "❌ 错误: Embedding 文件不存在: ${EMB_NPY_PATH}"
    echo "   请先执行 step3_text2emb.sh"
    exit 1
fi

cd "${SCRIPT_DIR}/scripts"

python train_rqvae.py \
    --data_path "${EMB_NPY_PATH}" \
    --ckpt_dir "${OUTPUT_CHECKPOINTS}/${DATASET_NAME}" \
    --device "${DEVICE_TRAIN_RQVAE}" \
    --lr ${TRAIN_LR} \
    --epochs ${TRAIN_EPOCHS} \
    --batch_size ${TRAIN_BATCH_SIZE} \
    --eval_step ${TRAIN_EVAL_STEP}

echo ""
echo "✅ Step 4 完成: RQ-VAE 训练完毕"
echo "   模型保存在: ${OUTPUT_CHECKPOINTS}/${DATASET_NAME}/"
echo ""
echo "⚠️  请将最佳模型路径填入 config.sh 的 RQVAE_CKPT_PATH"
echo "   例如: RQVAE_CKPT_PATH=\"${OUTPUT_CHECKPOINTS}/${DATASET_NAME}/xxxx/best_collision_model.pth\""
