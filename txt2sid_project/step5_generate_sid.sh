#!/bin/bash
# ============================================================================
# Step 5: 从 Embedding 生成 SID（含 Sinkhorn 消碰）
# 输入:  embedding .npy + RQ-VAE checkpoint
# 输出:  SID index .json
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "Step 5: Embedding → SID"
echo "=========================================="
echo "  设备:       ${DEVICE_GENERATE_SID}"
echo "  Embedding:  ${EMB_NPY_PATH}"
echo "  RQ-VAE:     ${RQVAE_CKPT_PATH}"
echo "  输出目录:   ${OUTPUT_SIDS}"
echo "=========================================="

# 检查输入
if [ ! -f "${EMB_NPY_PATH}" ]; then
    echo "❌ 错误: Embedding 文件不存在: ${EMB_NPY_PATH}"
    echo "   请先执行 step3_text2emb.sh"
    exit 1
fi

if [ -z "${RQVAE_CKPT_PATH}" ] || [ ! -f "${RQVAE_CKPT_PATH}" ]; then
    echo "❌ 错误: RQ-VAE checkpoint 不存在或未配置"
    echo "   请在 config.sh 中设置 RQVAE_CKPT_PATH"
    echo "   当前值: ${RQVAE_CKPT_PATH}"
    exit 1
fi

cd "${SCRIPT_DIR}/scripts"

python generate_sid.py \
    --ckpt_path "${RQVAE_CKPT_PATH}" \
    --data_path "${EMB_NPY_PATH}" \
    --output_dir "${OUTPUT_SIDS}" \
    --dataset_name "${DATASET_NAME}" \
    --device "${DEVICE_GENERATE_SID}" \
    --batch_size ${SID_BATCH_SIZE} \
    --max_collision_refine_steps ${SID_MAX_REFINE_STEPS} \
    --model_name "${EMB_MODEL_NAME}"

echo ""
echo "✅ Step 5 完成: SID 已生成"
echo "   输出文件: ${OUTPUT_SIDS}/${DATASET_NAME}.index-${EMB_MODEL_NAME}.json"
