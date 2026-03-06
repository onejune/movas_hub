#!/bin/bash

SOURCE_DIR="dnn"

# 设置目标目录（固定路径）
DEST_DIR="git_project/DeepForgeX/workshop/ms_dnn_model/$SOURCE_DIR/"

# 创建目标目录（如果不存在）
mkdir -p "$DEST_DIR"

# 使用 rsync 进行复制，排除 train_output 子目录
rsync -av --exclude='train_output/' --exclude='online_model/' --exclude='output/' --exclude='log/' --exclude='nohup.out' "$SOURCE_DIR/" "$DEST_DIR/"

# 检查 rsync 是否执行成功
if [ $? -eq 0 ]; then
    echo "✅ 文件复制成功。"
else
    echo "❌ 文件复制失败，请检查源目录或权限设置。"
fi
