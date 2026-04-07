#!/bin/bash
# dnn_ivr16_v2_slim - 特征精简实验
# 特征数: 341 -> 179 (删除冗余时间窗口，新增6个交叉特征)

source /mnt/workspace/walter.wan/utils/scripts/dnn_lib_common.sh

model_train
