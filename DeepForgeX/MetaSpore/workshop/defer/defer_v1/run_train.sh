#!/bin/bash
# DEFER v1 训练启动脚本
source /mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/utils/dnn_lib_common.sh

env_check
defer_train ./conf/config.yaml business_type
