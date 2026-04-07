#!/bin/bash
source /mnt/workspace/walter.wan/utils/scripts/dnn_lib_common.sh

# 参数: model_date sample_date [eval_keys] [conf_file]
model_validation "$1" "$2" "${3:-business_type}" "${4:-./conf/widedeep.yaml}"
