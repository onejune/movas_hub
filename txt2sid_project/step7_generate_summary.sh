#!/bin/bash
# ============================================================================
# Step 7: 生成汇总表格（文本 + Embedding + SID）
# 输出: CSV 和 JSON 格式的完整对照表
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "Step 7: 生成汇总表格"
echo "=========================================="

# 自动查找 SID 文件
SID_FILE=""
if [ -n "${EMB_MODEL_NAME}" ]; then
    SID_FILE="${OUTPUT_SIDS}/${DATASET_NAME}.index-${EMB_MODEL_NAME}.json"
fi
if [ ! -f "${SID_FILE}" ]; then
    SID_FILE="${OUTPUT_SIDS}/${DATASET_NAME}.index.json"
fi

echo "  数据文件:     ${ITEM_JSON_PATH}"
echo "  Embedding:    ${EMB_NPY_PATH}"
echo "  SID文件:      ${SID_FILE}"
echo "  输出目录:     ${OUTPUT_BASE}/summary/"
echo "=========================================="

# 构建参数
ARGS="--item_json ${ITEM_JSON_PATH}"
ARGS="${ARGS} --output ${OUTPUT_BASE}/summary/${DATASET_NAME}_summary"

if [ -f "${EMB_NPY_PATH}" ]; then
    ARGS="${ARGS} --emb_npy ${EMB_NPY_PATH}"
    echo "  ✓ 找到 Embedding 文件"
else
    echo "  ✗ 未找到 Embedding 文件（跳过）"
fi

if [ -f "${SID_FILE}" ]; then
    ARGS="${ARGS} --sid_json ${SID_FILE}"
    echo "  ✓ 找到 SID 文件"
else
    echo "  ✗ 未找到 SID 文件（跳过）"
fi

echo ""

cd "${SCRIPT_DIR}/scripts"
mkdir -p "${OUTPUT_BASE}/summary"

python generate_summary.py ${ARGS}

echo ""
echo "✅ Step 7 完成: 汇总表格已生成"
echo "   CSV:  ${OUTPUT_BASE}/summary/${DATASET_NAME}_summary.csv"
echo "   JSON: ${OUTPUT_BASE}/summary/${DATASET_NAME}_summary.json"
echo ""
echo "   CSV 可以直接用 Excel/WPS 打开查看"
