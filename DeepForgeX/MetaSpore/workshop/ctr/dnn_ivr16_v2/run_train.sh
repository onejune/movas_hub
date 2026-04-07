#!/bin/bash
# dnn_ivr16_v2 - 保守特征精简实验
# 特征数: 340 -> 253 (仅删除60d/90d/180d时间窗口和bucket特征，保留所有核心特征，新增6个交叉特征)

source /mnt/workspace/walter.wan/utils/scripts/dnn_lib_common.sh

model_train
