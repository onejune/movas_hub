#!/bin/bash
# DEFER 训练启动脚本
source /mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/utils/dnn_lib_common.sh

# 指定 DEFER 训练脚本
TRAINER_SCRIPT_PATH="./src/defer_trainFlow.py"

env_check
model_train ./conf/config.yaml business_type
