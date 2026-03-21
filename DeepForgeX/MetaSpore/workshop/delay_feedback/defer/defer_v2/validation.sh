#!/bin/bash
# DEFER v2 验证脚本
# 用法: ./validation.sh <model_date> <sample_date> [eval_keys]
# 示例: ./validation.sh 2026-02-17 2026-02-18 business_type
source /mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/utils/dnn_lib_common.sh

# 指定 DEFER 训练脚本
TRAINER_SCRIPT_PATH="./src/defer_trainFlow.py"

model_validation $1 $2 ${3:-business_type} ./conf/config.yaml
