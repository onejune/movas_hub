#!/bin/bash
# ============================================================================
# Step 6: 端到端 文本 → SID（跳过分步，直接从文本到SID）
# 需要已有训练好的 RQ-VAE 模型
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "Step 6: 端到端 文本 → SID"
echo "=========================================="
echo "  设备:     ${DEVICE_TEXT2SID}"
echo "  Qwen:     ${QWEN_MODEL_PATH}"
echo "  RQ-VAE:   ${RQVAE_CKPT_PATH}"
echo "=========================================="

# 检查 RQ-VAE
if [ -z "${RQVAE_CKPT_PATH}" ] || [ ! -f "${RQVAE_CKPT_PATH}" ]; then
    echo "❌ 错误: RQ-VAE checkpoint 不存在或未配置"
    echo "   请在 config.sh 中设置 RQVAE_CKPT_PATH"
    exit 1
fi

cd "${SCRIPT_DIR}/scripts"

# ======== 模式选择（取消注释你想要的模式）========

# --- 模式1: 单条文本 ---
python text2sid.py \
    --qwen_path "${QWEN_MODEL_PATH}" \
    --rqvae_ckpt "${RQVAE_CKPT_PATH}" \
    --device "${DEVICE_TEXT2SID}" \
    --text "Barbie Doll Pink Princess Dress for Girls Age 3-12"

# --- 模式2: 批量处理 ---
# python text2sid.py \
#     --qwen_path "${QWEN_MODEL_PATH}" \
#     --rqvae_ckpt "${RQVAE_CKPT_PATH}" \
#     --device "${DEVICE_TEXT2SID}" \
#     --item_json "${ITEM_JSON_PATH}" \
#     --output_file "${OUTPUT_SIDS}/${DATASET_NAME}.text2sid.json" \
#     --qwen_batch_size ${TEXT2SID_BATCH_SIZE} \
#     --evaluate

# --- 模式3: 交互模式 ---
# python text2sid.py \
#     --qwen_path "${QWEN_MODEL_PATH}" \
#     --rqvae_ckpt "${RQVAE_CKPT_PATH}" \
#     --device "${DEVICE_TEXT2SID}" \
#     --interactive
