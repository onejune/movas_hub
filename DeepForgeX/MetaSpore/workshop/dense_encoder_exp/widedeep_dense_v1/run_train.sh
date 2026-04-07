#!/bin/bash

source /mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/utils/scripts/dnn_lib_common.sh

TRAINER_SCRIPT_PATH="./src/dnn_trainFlow.py"

env_check
model_train ./conf/config.yaml business_type
