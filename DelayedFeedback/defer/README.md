# Delayed Feedback Modeling (DEFER)

Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling

## 项目结构

```
defer/
├── src_tf_github/          # TensorFlow 实现 (论文原版代码)
│   └── src/
│       ├── data.py         # 数据加载与预处理
│       ├── models.py       # 模型定义
│       ├── loss.py         # 损失函数 (FNW, FNC, DFM, ES-DFM, WinAdapt)
│       ├── main.py         # 训练入口
│       ├── pretrain.py     # 预训练逻辑
│       ├── stream_train_test.py  # 流式训练测试
│       ├── metrics.py      # 评估指标
│       ├── test.py         # 测试脚本
│       └── utils.py        # 工具函数
│
├── src_pytorch_v2/         # PyTorch 实现
│   ├── data.py             # Parquet 数据加载, 249 特征
│   ├── models.py           # DeferModel, WinAdapt 4-head 输出
│   ├── loss.py             # 损失函数实现
│   └── train.py            # 训练入口
│
├── src_pytorch_v5/         # PyTorch 实现 (推荐, 最新)
│   ├── data.py             # 数据加载, 支持 DFM/ES-DFM 字段
│   ├── models.py           # DeferModel, ES-DFM 两阶段模型
│   ├── loss.py             # 完整损失函数 (含 esdfm_pretrain_loss)
│   └── train.py            # 训练入口, 支持 ES-DFM 两阶段训练
│
├── scripts/
│   ├── preprocess_v2.py    # 数据预处理 (生成 data_v2)
│   └── preprocess_v3.py    # 数据预处理 (生成 data_v4, 含 DFM 字段)
│
├── convert_data.py         # 原始数据转换
├── convert_data_fast.py    # 快速数据转换 (大文件优化)
├── download_data.sh        # 下载公开数据集
└── run_serial.sh           # 串行训练脚本
```

## 数据处理

### 1. 下载原始数据

```bash
bash download_data.sh
```

### 2. 数据转换

原始 txt 格式转换为中间格式：

```bash
python convert_data.py --input data/criteo_data.txt --output data/converted/
# 或使用快速版本 (大文件推荐)
python convert_data_fast.py --input data/criteo_data.txt --output data/converted/
```

### 3. 生成 Parquet 格式

**data_v2 (基础版本):**
```bash
python scripts/preprocess_v2.py \
    --input data/converted/ \
    --output data_v2/ \
    --time_windows 24,48,72,168
```

**data_v4 (含 DFM/ES-DFM 字段, 推荐):**
```bash
python scripts/preprocess_v3.py \
    --input data/converted/ \
    --output data_v4/ \
    --time_windows 24,48,72,168
```

data_v4 额外包含：
- `delay_time` - 转化延迟时间 (正样本为实际延迟, 负样本为 168h)
- `elapsed_time` - 观察窗口时间
- `tn_label` - True Negative 标签
- `dp_label` - Delayed Positive 标签
- `pos_label` - 窗口内正样本标签

## 模型训练

### PyTorch v5 (推荐)

```bash
cd src_pytorch_v5

# Vanilla (基础模型)
python train.py --method vanilla --epochs 1 --batch_size 4096

# Oracle (带转化标签, 上界参考)
python train.py --method oracle --epochs 1 --batch_size 4096

# WinAdapt (窗口自适应)
python train.py --method winadapt --epochs 1 --batch_size 4096

# DFM (Delayed Feedback Model)
python train.py --method dfm --epochs 1 --batch_size 4096

# ES-DFM (两阶段训练)
python train.py --method esdfm \
    --esdfm_pretrain_epochs 1 \
    --esdfm_finetune_epochs 1 \
    --batch_size 4096
```

支持的方法：
- `vanilla` - 基础 CTR 预测
- `oracle` - 带转化标签 (上界参考)
- `fnw` - Fake Negative Weighted
- `fnc` - Fake Negative Calibration  
- `dfm` - Delayed Feedback Model (生存分析)
- `esdfm` - ES-DFM (两阶段重要性加权)
- `winadapt` - Window Adapter

### TensorFlow 原版

```bash
cd src_tf_github/src
python main.py --method oracle --epochs 1
```

## 实验结果

| 方法 | AUC | PR-AUC | LogLoss | 备注 |
|------|-----|--------|---------|------|
| **ES-DFM** | **0.8548** | **0.3247** | **0.2028** | 两阶段训练, 最佳 |
| WinAdapt | 0.8526 | 0.3178 | 0.2031 | |
| Oracle | 0.8515 | 0.3163 | 0.2038 | 理论上界 |
| DFM | 0.8502 | 0.3124 | 0.2045 | |
| FNC | 0.6911 | 0.2014 | 0.2987 | |
| FNW | 0.6746 | 0.1927 | 0.3322 | |
| Vanilla | 0.6438 | 0.1826 | 0.3761 | |

训练配置: 1 epoch, batch_size=4096, lr=0.001, embed_dim=8, CPU

## ES-DFM 两阶段训练

ES-DFM 通过重要性加权解决延迟反馈问题：

**Stage 1 - Pretrain tn/dp classifiers:**
- 训练 `tn_logits`: 区分真负样本 vs 延迟正样本
- 训练 `dp_logits`: 预测延迟正样本概率

**Stage 2 - Finetune CVR with importance weighting:**
- 固定 tn/dp 分类器 (detach)
- 使用重要性权重训练 CVR: `w = z + (1-z) * w_neg`
- 其中 `w_neg = tn_prob / (tn_prob + dp_prob)`

关键修复：负样本的 `delay_time` 应设为观察窗口长度 (168h)，而非 0。

## WinAdapt 设计

WinAdapt 使用 4 个输出头，分别预测不同时间窗口的转化概率：
- 24h, 48h, 72h, 168h

训练流程：
1. Vanilla 预训练 → 学习特征表示
2. Oracle 训练 → 加入转化标签作为特征
3. WinAdapt → 冻结主模型，训练 Window Adapter 模拟 Oracle

## 参考

- 论文: Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling
- 原始代码: https://github.com/ThyrixYang/es_dfm
