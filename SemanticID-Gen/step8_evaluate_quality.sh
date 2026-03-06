#!/bin/bash
# ============================================================================
# Step 8: SID 质量评估
# 输入: embedding .npy + SID .json
# 输出: 质量评估报告（控制台 + JSON）
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "Step 8: SID 质量评估"
echo "=========================================="

# 自动查找 SID 文件
SID_FILE=""
if [ -n "${EMB_MODEL_NAME}" ]; then
    SID_FILE="${OUTPUT_SIDS}/${DATASET_NAME}.index-${EMB_MODEL_NAME}.json"
fi
if [ ! -f "${SID_FILE}" ]; then
    SID_FILE="${OUTPUT_SIDS}/${DATASET_NAME}.index.json"
fi

echo "  Embedding:    ${EMB_NPY_PATH}"
echo "  SID文件:      ${SID_FILE}"
echo "=========================================="

# 检查输入
if [ ! -f "${EMB_NPY_PATH}" ]; then
    echo "❌ 错误: Embedding 文件不存在: ${EMB_NPY_PATH}"
    exit 1
fi

if [ ! -f "${SID_FILE}" ]; then
    echo "❌ 错误: SID 文件不存在: ${SID_FILE}"
    echo "   请先执行 step5_generate_sid.sh"
    exit 1
fi

cd "${SCRIPT_DIR}/scripts"
mkdir -p "${OUTPUT_BASE}/evaluation"

OUTPUT_JSON="${OUTPUT_BASE}/evaluation/${DATASET_NAME}_quality.json"

python evaluate_sid_quality.py \
    --emb_npy "${EMB_NPY_PATH}" \
    --sid_json "${SID_FILE}" \
    --output "${OUTPUT_JSON}" \
    --K 10

echo ""
echo "✅ Step 8 完成: 质量评估已完成"
echo "   评估报告: ${OUTPUT_JSON}"
