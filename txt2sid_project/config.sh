#!/bin/bash
# ============================================================================
# 统一配置文件 - 只需要改这个文件，所有步骤自动生效
# ============================================================================

# ======================== 设备配置 ========================
# 每个步骤可以单独设置 CPU 或 GPU
# 填 "cpu" 或 "cuda:0" 或 "cuda:1" 等

DEVICE_TEXT2EMB="cpu"          # Step3: 文本转Embedding（7B模型需要大显存，没GPU就用cpu）
DEVICE_TRAIN_RQVAE="cpu"      # Step4: 训练RQ-VAE（推荐GPU，cpu也能跑但慢）
DEVICE_GENERATE_SID="cpu"     # Step5: 生成SID（轻量，cpu也很快）
DEVICE_TEXT2SID="cpu"          # Step6: 端到端文本→SID

# ======================== 模型路径 ========================
# Qwen 模型路径（Step2下载后自动填好，也可以手动改）
QWEN_MODEL_PATH="/mnt/workspace/models/Qwen/Qwen2.5-7B"

# RQ-VAE checkpoint 路径（Step4训练完后自动生成，也可以手动指定）
# 训练完成后会打印路径，填到这里
RQVAE_CKPT_PATH=""

# ======================== 数据路径 ========================
# 输入数据文件（item.json格式）
ITEM_JSON_PATH="/mnt/workspace/MiniOneRec/data/Amazon/index/Toys_and_Games_10k.item.json"

# 数据集名称（用于输出文件命名）
DATASET_NAME="Toys_and_Games_10k"

# ======================== 输出目录 ========================
# 所有中间结果和最终结果都保存在这里
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${PIPELINE_DIR}/output"

OUTPUT_EMBEDDINGS="${OUTPUT_BASE}/embeddings"    # Step3输出: embedding .npy文件
OUTPUT_CHECKPOINTS="${OUTPUT_BASE}/checkpoints"  # Step4输出: RQ-VAE模型
OUTPUT_SIDS="${OUTPUT_BASE}/sids"                # Step5输出: SID .json文件

# ======================== 训练参数 ========================
# RQ-VAE 训练参数
TRAIN_EPOCHS=5000
TRAIN_BATCH_SIZE=2048
TRAIN_LR=1e-3
TRAIN_EVAL_STEP=50

# ======================== 其他参数 ========================
# 文本转Embedding
EMB_BATCH_SIZE=8               # CPU建议8，GPU可以32-64
EMB_MAX_LEN=512                # 文本最大token长度
EMB_MODEL_NAME="qwen"         # 模型名（用于文件命名）

# 生成SID
SID_BATCH_SIZE=64
SID_MAX_REFINE_STEPS=20

# 端到端
TEXT2SID_BATCH_SIZE=4

# ======================== 自动创建输出目录 ========================
mkdir -p "${OUTPUT_EMBEDDINGS}"
mkdir -p "${OUTPUT_CHECKPOINTS}"
mkdir -p "${OUTPUT_SIDS}"

# ======================== 自动推导路径 ========================
# Embedding文件路径（Step3输出，Step4/5输入）
EMB_NPY_PATH="${OUTPUT_EMBEDDINGS}/${DATASET_NAME}.emb-${EMB_MODEL_NAME}-td.npy"

echo "=========================================="
echo "Pipeline 配置已加载"
echo "=========================================="
echo "  设备配置:"
echo "    Text2Emb:      ${DEVICE_TEXT2EMB}"
echo "    Train RQ-VAE:  ${DEVICE_TRAIN_RQVAE}"
echo "    Generate SID:  ${DEVICE_GENERATE_SID}"
echo "    Text2SID:      ${DEVICE_TEXT2SID}"
echo ""
echo "  数据:"
echo "    输入数据:      ${ITEM_JSON_PATH}"
echo "    数据集名:      ${DATASET_NAME}"
echo ""
echo "  输出目录:"
echo "    Embeddings:    ${OUTPUT_EMBEDDINGS}"
echo "    Checkpoints:   ${OUTPUT_CHECKPOINTS}"
echo "    SIDs:          ${OUTPUT_SIDS}"
echo "=========================================="
