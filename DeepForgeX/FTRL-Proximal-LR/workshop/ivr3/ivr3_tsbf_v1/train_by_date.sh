#!/bin/bash

source /mnt/workspace/walter.wan/utils/lib_common.sh

# 参数校验
if [ $# -ne 2 ]; then
    echo "用法: $0 <开始日期> <结束日期>"
    echo "示例: $0 2024-08-30 2025-02-04"
    exit 1
fi

train_by_date $1 $2
