#!/bin/bash
source /mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/utils/dnn_lib_common.sh
TRAINER_SCRIPT_PATH="./src/defer_trainFlow.py"
env_check
model_train ./conf/config.yaml business_type
