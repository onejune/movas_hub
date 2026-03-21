#!/bin/bash
# DEFER v2 训练启动脚本
# 长时间窗口 (24/48/72 h), 模型: WinAdaptDNN
source /mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/utils/dnn_lib_common.sh

# 指定 DEFER 训练脚本
TRAINER_SCRIPT_PATH="./src/defer_trainFlow.py"

env_check
model_train ./conf/config.yaml business_type
