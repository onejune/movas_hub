#!/bin/bash
# ============================================================================
# Step 2: 下载文本编码模型
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=========================================="
echo "Step 2: 下载文本编码模型"
echo "=========================================="
echo "目标模型路径: ${QWEN_MODEL_PATH}"

# 检查模型是否已存在
if [ -d "${QWEN_MODEL_PATH}" ] && [ "$(ls -A ${QWEN_MODEL_PATH} 2>/dev/null)" ]; then
    echo "✅ 模型已存在: ${QWEN_MODEL_PATH}"
    echo "   如需重新下载，请先删除该目录"
    exit 0
fi

# 从 ModelScope 下载（国内推荐）
# 根据路径自动推断模型名称
MODEL_NAME=$(basename "$(dirname "${QWEN_MODEL_PATH}")")/$(basename "${QWEN_MODEL_PATH}")
CACHE_DIR=$(dirname "$(dirname "${QWEN_MODEL_PATH}")")

echo "下载模型: ${MODEL_NAME}"
echo "保存到:   ${CACHE_DIR}"

python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('${MODEL_NAME}', cache_dir='${CACHE_DIR}')
print('下载完成:', model_dir)
"

echo ""
echo "✅ Step 2 完成: 模型下载成功"
echo "   模型路径: ${QWEN_MODEL_PATH}"

# ==========================================================================
# 如果想下载其他模型，修改 config.sh 中的 QWEN_MODEL_PATH，例如:
#
#   小模型（快速测试）:
#   QWEN_MODEL_PATH="/mnt/workspace/models/Qwen/Qwen2.5-0.5B"
#
#   大模型（效果更好）:
#   QWEN_MODEL_PATH="/mnt/workspace/models/Qwen/Qwen2.5-14B"
#
# 也可以用 HuggingFace 下载（国际）:
#   export HF_ENDPOINT=https://hf-mirror.com  # 国内加速
#   pip install huggingface_hub
#   python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download('Qwen/Qwen2.5-7B', cache_dir='/mnt/workspace/models')
#   "
# ==========================================================================
