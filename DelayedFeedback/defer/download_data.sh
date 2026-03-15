#!/bin/bash
# =============================================================================
# Criteo 延迟反馈数据集下载脚本
# =============================================================================

set -e

DATA_DIR="${DATA_DIR:-/mnt/workspace/walter.wan/open_research/defer/data}"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "Criteo 延迟反馈数据集下载"
echo "=============================================="
echo ""
echo "数据集信息:"
echo "  - 60 天数据"
echo "  - ~1590 万样本"
echo "  - ~360 万转化"
echo "  - CVR ≈ 22.69%"
echo ""
echo "下载地址:"
echo "  https://labs.criteo.com/2013/12/conversion-logs-dataset/"
echo ""
echo "注意: 由于版权原因，需要手动下载数据集"
echo ""
echo "下载步骤:"
echo "  1. 访问 https://labs.criteo.com/2013/12/conversion-logs-dataset/"
echo "  2. 填写表格获取下载链接"
echo "  3. 下载 criteo_conversion_logs.tar.gz"
echo "  4. 解压到 $DATA_DIR 目录"
echo ""
echo "或者使用以下命令 (如果你已有下载链接):"
echo "  wget -O criteo_conversion_logs.tar.gz '<your_download_link>'"
echo "  tar -xzf criteo_conversion_logs.tar.gz"
echo "  mv criteo_conversion_logs/data.txt $DATA_DIR/criteo_data.txt"
echo ""
echo "=============================================="

# 检查是否已有数据
if [[ -f "$DATA_DIR/criteo_data.txt" ]]; then
    echo "数据文件已存在: $DATA_DIR/criteo_data.txt"
    echo "文件大小: $(ls -lh "$DATA_DIR/criteo_data.txt" | awk '{print $5}')"
    echo "行数: $(wc -l < "$DATA_DIR/criteo_data.txt")"
else
    echo "数据文件不存在，请按上述步骤下载"
fi
