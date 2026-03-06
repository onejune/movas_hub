#!/bin/bash

# 定义输入和输出目录
input_dir="../ruf/sample"
output_dir="./sample"

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 遍历所有 CSV 文件
for file in "$input_dir"/sample*; do
    # 获取文件名（如 2024-10-01.csv）
    filename=$(basename "$file")
    
    # 构建输出路径
    output_file="$output_dir/$filename"
    
    # 使用 awk 筛选第 13 列为 "COM.ZZKKO" 的行
    awk -F'\002' '$13 == "COM.ZZKKO" {print}' "$file" > "$output_file"
done