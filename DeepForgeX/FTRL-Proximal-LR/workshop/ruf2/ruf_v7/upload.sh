#!/bin/bash

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_date> <end_date> (format: YYYY-MM-DD)"
    exit 1
fi

# 获取日期范围
start_date=$1
end_date=$2

# 确保开始日期 <= 结束日期
if [[ "$start_date" > "$end_date" ]]; then
    echo "Error: Start date must be less than or equal to end date."
    exit 1
fi

# OSS 目标路径（根据实际情况修改）
oss_path="oss://spark-ml-train-new/dsp_algo/ivr/model/ruf2_v2"

# 遍历日期范围
current_date="$start_date"
while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
    # 构建源文件路径和目标文件名
    src_file="train_output/base.$current_date"
    dest_file="$oss_path/model_file_$current_date"

    # 检查文件是否存在
    if [ -f "$src_file" ]; then
        echo "Uploading $src_file to $dest_file..."
        ossutil cp "$src_file" "$dest_file"
        if [ $? -eq 0 ]; then
            echo "Success: $src_file uploaded."
        else
            echo "Error: Failed to upload $src_file."
        fi
    else
        echo "Warning: File not found - $src_file"
    fi

    # 更新日期（使用 date 命令递增）
    current_date=$(date -I -d "$current_date + 1 day")
done

echo "All files processed."
