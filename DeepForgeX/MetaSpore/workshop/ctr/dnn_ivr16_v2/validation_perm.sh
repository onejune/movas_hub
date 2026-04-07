#!/bin/bash
# Permutation Importance 验证脚本 (auto-generated)
# 用法: ./validation_perm.sh <model_date> <sample_date> <shuffle_feature> <eval_keys>

source /mnt/workspace/walter.wan/utils/scripts/dnn_lib_common.sh
model_validation "$1" "$2" "$4" "./conf/widedeep.yaml" "$3"
