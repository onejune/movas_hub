#!/bin/bash

CURRENT_DIR_NAME=$(basename "$PWD")
echo "[INFO] 当前目录名称: $CURRENT_DIR_NAME"

# 修复点：用 -- 明确结束 grep 的选项解析，后面的内容全视为 pattern
# 同时用双引号包裹整个匹配字符串
PIDS=$(ps aux | grep "python" | grep -- "--name $CURRENT_DIR_NAME" | grep -v grep | awk '{print $2}')
COMMAND_LINES=$(ps aux | grep "python" | grep -- "--name $CURRENT_DIR_NAME" | grep -v grep)

if [ -z "$PIDS" ]; then
    echo "[INFO] 未找到包含 '--name $CURRENT_DIR_NAME' 的运行中进程。"
else
    echo "[INFO] 找到以下匹配的进程："
    while IFS= read -r line; do
        echo "  ➤ $line"
    done <<< "$COMMAND_LINES"

    for PID in $PIDS; do
        echo "[INFO] 正在终止进程 PID=$PID ..."
        if kill -9 "$PID" 2>/dev/null; then
            echo "[SUCCESS] 成功终止 PID=$PID"
        else
            echo "[ERROR] 无法终止 PID=$PID（可能已退出）"
        fi
    done
fi

echo "[INFO] check 包含 train 的进程"
ps -aux | grep train