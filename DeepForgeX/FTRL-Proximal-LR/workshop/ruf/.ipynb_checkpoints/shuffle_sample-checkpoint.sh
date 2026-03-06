#!/bin/bash

# 检查是否传入了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_date> <end_date>"
    echo "Example: $0 2026-04-01 2026-04-05"
    exit 1
fi

# 获取起始和结束日期
start_date="$1"
end_date="$2"

# 检查日期格式是否为 YYYY-MM-DD
if ! [[ "$start_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Error: Invalid start date format. Expected YYYY-MM-DD."
    exit 1
fi

if ! [[ "$end_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Error: Invalid end date format. Expected YYYY-MM-DD."
    exit 1
fi

# 设置当前日期为起始日期
current_date="$start_date"

# 循环处理每个日期
while [ "$current_date" != "$end_date" ]; do
    # 构造文件名
    filename="sample_${current_date}"

    # 检查文件是否存在
    if [ -f "./sample/$filename" ]; then
        echo "Shuffling $filename..."
        shuf "./sample/$filename" -o "./sample/$filename"
    else
        echo "File not found: $filename"
    fi

    # 递增日期
    current_date=$(date -d "$current_date + 1 day" +"%Y-%m-%d")
done

# 处理最后一个日期
filename="sample_${end_date}"
if [ -f "./sample/$filename" ]; then
    echo "Shuffling $filename..."
    shuf "./sample/$filename" -o "./sample/$filename"
else
    echo "File not found: $filename"
fi

echo "All files in the specified date range have been shuffled."