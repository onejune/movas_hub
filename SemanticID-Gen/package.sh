#!/bin/bash
# ============================================================================
# 打包脚本：生成干净的 pipeline 压缩包用于分发
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

PACKAGE_NAME="sid_pipeline_v1.0"
OUTPUT_DIR="${SCRIPT_DIR}/../packages"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "打包 SID Pipeline"
echo "=========================================="

# 创建临时目录
TEMP_DIR="/tmp/${PACKAGE_NAME}"
rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

# 复制文件
echo "复制文件..."
cp -r "${SCRIPT_DIR}" "${TEMP_DIR}/"

# 清理不需要的文件
echo "清理临时文件..."
cd "${TEMP_DIR}/pipeline"

# 删除输出目录
rm -rf output/

# 删除 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# 删除 IDE 配置
rm -rf .vscode/ .idea/

# 删除 git 相关
rm -rf .git/ .gitignore

# 创建空的输出目录结构
mkdir -p output/embeddings
mkdir -p output/checkpoints
mkdir -p output/sids
mkdir -p output/summary
mkdir -p output/evaluation

# 在每个目录放一个说明文件
echo "此目录用于存放 Embedding 文件 (.npy)" > output/embeddings/README.txt
echo "此目录用于存放 RQ-VAE 模型文件 (.pth)" > output/checkpoints/README.txt
echo "此目录用于存放 SID 文件 (.json)" > output/sids/README.txt
echo "此目录用于存放汇总表格 (.csv/.json)" > output/summary/README.txt
echo "此目录用于存放质量评估报告 (.json)" > output/evaluation/README.txt

# 打包
echo "压缩打包..."
mkdir -p "${OUTPUT_DIR}"
cd "${TEMP_DIR}"
tar -czf "${OUTPUT_DIR}/${PACKAGE_NAME}_${TIMESTAMP}.tar.gz" pipeline/

# 创建一个不带时间戳的最新版本
cp "${OUTPUT_DIR}/${PACKAGE_NAME}_${TIMESTAMP}.tar.gz" "${OUTPUT_DIR}/${PACKAGE_NAME}_latest.tar.gz"

# 清理临时目录
rm -rf "${TEMP_DIR}"

echo ""
echo "✅ 打包完成！"
echo "   文件位置: ${OUTPUT_DIR}/${PACKAGE_NAME}_${TIMESTAMP}.tar.gz"
echo "   最新版本: ${OUTPUT_DIR}/${PACKAGE_NAME}_latest.tar.gz"
echo ""
echo "文件大小:"
ls -lh "${OUTPUT_DIR}/${PACKAGE_NAME}_latest.tar.gz"
echo ""
echo "=========================================="
echo "下一步：上传到 OSS"
echo "=========================================="
echo ""
echo "1. 使用 ossutil 上传："
echo "   ossutil cp ${OUTPUT_DIR}/${PACKAGE_NAME}_latest.tar.gz oss://your-bucket/path/"
echo ""
echo "2. 或使用 DSW 内置的 OSS 工具"
echo ""
echo "3. 上传后获取下载链接："
echo "   https://your-bucket.oss-cn-region.aliyuncs.com/path/${PACKAGE_NAME}_latest.tar.gz"
echo ""
