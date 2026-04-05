#!/bin/bash
#
# Permutation Importance 运行脚本
#
# 用法:
#   ./run_permutation.sh <project_dir> <model_date> <sample_date> [features]
#
# 示例:
#   ./run_permutation.sh ./dnn_ivr16_v1 2026-03-02 2026-03-03
#   ./run_permutation.sh ./dnn_ivr16_v1 2026-03-02 2026-03-03 "country,adx,city"
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="${PYTHON_ENV:-/root/anaconda3/envs/spore/bin/python}"

PROJECT_DIR="$1"
MODEL_DATE="$2"
SAMPLE_DATE="$3"
FEATURES="$4"

if [ -z "$PROJECT_DIR" ] || [ -z "$MODEL_DATE" ] || [ -z "$SAMPLE_DATE" ]; then
    echo "用法: $0 <project_dir> <model_date> <sample_date> [features]"
    echo ""
    echo "示例:"
    echo "  $0 ./dnn_ivr16_v1 2026-03-02 2026-03-03"
    echo "  $0 ./dnn_ivr16_v1 2026-03-02 2026-03-03 'country,adx,city'"
    exit 1
fi

# 转换为绝对路径
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

echo "============================================================"
echo "Permutation Importance 分析"
echo "============================================================"
echo "项目目录: $PROJECT_DIR"
echo "模型日期: $MODEL_DATE"
echo "样本日期: $SAMPLE_DATE"
echo "特征: ${FEATURES:-全部}"
echo ""

# 构建命令
CMD="$PYTHON_ENV $SCRIPT_DIR/permutation_importance.py"
CMD="$CMD --project_dir $PROJECT_DIR"
CMD="$CMD --model_date $MODEL_DATE"
CMD="$CMD --sample_date $SAMPLE_DATE"

if [ -n "$FEATURES" ]; then
    CMD="$CMD --features $FEATURES"
fi

echo "执行: $CMD"
echo ""

exec $CMD
