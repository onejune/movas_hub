#!/bin/bash
# VW FTRL-LR Training Script
# Usage: bash train.sh [TRAIN_DATE] [VAL_DATE]
# Example: bash train.sh 2026-02-18 2026-02-19

set -e

TRAIN_DATE=${1:-"2026-02-18"}
VAL_DATE=${2:-"2026-02-19"}

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKSHOP_DIR="$(cd "$(dirname "$0")" && pwd)"

# 数据路径 (CSV 格式，需要先用 csv2vw 转换)
CSV_ROOT="/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v7/csv"
VW_DATA_ROOT="/mnt/workspace/walter.wan/vw_data/ivr16_aert_v1"

SCHEMA="${WORKSHOP_DIR}/conf/combine_schema"
OUTPUT_DIR="${WORKSHOP_DIR}/output"

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "VW FTRL-LR Training"
echo "=============================================="
echo "Train date: ${TRAIN_DATE}"
echo "Val date: ${VAL_DATE}"
echo "Schema: ${SCHEMA}"
echo ""

# Check if VW is installed
if ! command -v vw &> /dev/null; then
    echo "Error: vowpal-wabbit not installed. Run: apt-get install -y vowpal-wabbit"
    exit 1
fi

# Check if csv2vw is compiled
CSV2VW="${PROJECT_ROOT}/tools/csv2vw"
if [ ! -f "${CSV2VW}" ]; then
    echo "Compiling csv2vw..."
    gcc -O3 -o ${CSV2VW} ${PROJECT_ROOT}/tools/csv2vw.c
fi

# Prepare train data
TRAIN_VW="${VW_DATA_ROOT}/train_${TRAIN_DATE}.vw"
if [ ! -f "${TRAIN_VW}" ]; then
    echo "[Step 1] Converting training data..."
    TRAIN_CSV="${CSV_ROOT}/part=${TRAIN_DATE}"
    cat ${TRAIN_CSV}/*.csv | ${CSV2VW} ${SCHEMA} > ${TRAIN_VW}
    echo "  -> $(wc -l < ${TRAIN_VW}) lines"
else
    echo "[Step 1] Using cached training data: ${TRAIN_VW}"
fi

# Prepare val data
VAL_VW="${VW_DATA_ROOT}/val_${VAL_DATE}.vw"
if [ ! -f "${VAL_VW}" ]; then
    echo "[Step 2] Converting validation data..."
    VAL_CSV="${CSV_ROOT}/part=${VAL_DATE}"
    cat ${VAL_CSV}/*.csv | ${CSV2VW} ${SCHEMA} > ${VAL_VW}
    echo "  -> $(wc -l < ${VAL_VW}) lines"
else
    echo "[Step 2] Using cached validation data: ${VAL_VW}"
fi

# Training
echo ""
echo "[Step 3] Training VW FTRL..."
MODEL_FILE="${OUTPUT_DIR}/model_${TRAIN_DATE}.vw"

time vw --ftrl \
    --ftrl_alpha 0.1 \
    --ftrl_beta 1 \
    --l1 1 \
    --l2 300 \
    -b 24 \
    --loss_function logistic \
    --link logistic \
    -d ${TRAIN_VW} \
    -f ${MODEL_FILE} \
    --cache_file ${OUTPUT_DIR}/train.cache \
    2>&1 | tail -10

# Prediction
echo ""
echo "[Step 4] Predicting..."
PRED_FILE="${OUTPUT_DIR}/pred_${VAL_DATE}.txt"
vw -t -i ${MODEL_FILE} -p ${PRED_FILE} ${VAL_VW} 2>/dev/null

# Evaluation
echo ""
echo "[Step 5] Evaluating..."
python ${PROJECT_ROOT}/tools/evaluate.py -d ${VAL_VW} -p ${PRED_FILE}

echo ""
echo "=============================================="
echo "Training completed!"
echo "Model: ${MODEL_FILE}"
echo "Predictions: ${PRED_FILE}"
echo "=============================================="
