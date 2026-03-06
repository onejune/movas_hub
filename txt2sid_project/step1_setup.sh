#!/bin/bash
# ============================================================================
# Step 1: 创建环境 + 安装依赖
# ============================================================================

echo "=========================================="
echo "Step 1: 安装环境和依赖"
echo "=========================================="

# 创建 conda 环境（如果还没有）
conda create -n minionerec-sid python=3.11 -y 2>/dev/null || true
conda activate minionerec-sid

# 安装依赖
pip install torch numpy tqdm transformers modelscope scikit-learn
pip install tiktoken protobuf sentencepiece

echo ""
echo "✅ Step 1 完成: 环境和依赖安装成功"
echo "   请执行: conda activate minionerec-sid"
