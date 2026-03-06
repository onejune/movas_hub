#!/bin/bash
# ============================================================================
# Step 3: 商品文本 → Embedding 向量
# 输入:  item.json（商品标题+描述）
# 输出:  embeddings .npy 文件
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "Step 3: 文本 → Embedding"
echo "=========================================="
echo "  设备:     ${DEVICE_TEXT2EMB}"
echo "  模型:     ${QWEN_MODEL_PATH}"
echo "  输入:     ${ITEM_JSON_PATH}"
echo "  输出:     ${EMB_NPY_PATH}"
echo "=========================================="

# 检查输入文件
if [ ! -f "${ITEM_JSON_PATH}" ]; then
    echo "❌ 错误: 输入文件不存在: ${ITEM_JSON_PATH}"
    exit 1
fi

# 检查模型
if [ ! -d "${QWEN_MODEL_PATH}" ]; then
    echo "❌ 错误: 模型不存在: ${QWEN_MODEL_PATH}"
    echo "   请先执行 step2_download_model.sh"
    exit 1
fi

cd "${SCRIPT_DIR}/scripts"

python text2emb.py \
    --item_json "${ITEM_JSON_PATH}" \
    --model_path "${QWEN_MODEL_PATH}" \
    --output "${EMB_NPY_PATH}" \
    --device "${DEVICE_TEXT2EMB}" \
    --batch_size ${EMB_BATCH_SIZE} \
    --max_len ${EMB_MAX_LEN} \
    --plm_name "${EMB_MODEL_NAME}"

echo ""
echo "✅ Step 3 完成: Embedding 已保存"
echo "   输出文件: ${EMB_NPY_PATH}"
