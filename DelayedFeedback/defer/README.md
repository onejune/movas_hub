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
├── src_pytorch_v2/         # PyTorch 实现 (推荐)
│   ├── data.py             # Parquet 数据加载, 249 特征
│   ├── models.py           # DeferModel, WinAdapt 4-head 输出
│   ├── loss.py             # 损失函数实现
│   └── train.py            # 训练入口
│
├── scripts/
│   └── preprocess_v2.py    # 数据预处理 (生成 parquet)
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

### 3. 生成 Parquet 格式 (PyTorch v2 使用)

```bash
python scripts/preprocess_v2.py \
    --input data/converted/ \
    --output data_v2/ \
    --time_windows 24,48,72,168
```

输出文件：
- `data_v2/train.parquet` - 训练集 (~1100 万样本)
- `data_v2/test.parquet` - 测试集 (~75 万样本)
- `data_v2/feature_cols.json` - 特征列配置

## 模型训练

### PyTorch v2 (推荐)

```bash
cd src_pytorch_v2

# Vanilla (基础模型)
python train.py --method vanilla --epochs 1 --batch_size 4096

# Oracle (带转化标签)
python train.py --method oracle --epochs 1 --batch_size 4096

# WinAdapt (窗口自适应)
python train.py --method winadapt --epochs 1 --batch_size 4096
```

支持的方法：
- `vanilla` - 基础 CTR 预测
- `oracle` - 带转化标签 (上界参考)
- `fnw` - Fake Negative Weighted
- `fnc` - Fake Negative Calibration  
- `dfm` - Delayed Feedback Model
- `esdfm` - ES-DFM (重要性加权)
- `winadapt` - Window Adapter (本项目重点)

### TensorFlow 原版

```bash
cd src_tf_github/src
python main.py --method oracle --epochs 1
```

## 实验结果

| 方法 | AUC | PR-AUC | LogLoss |
|------|-----|--------|---------|
| Vanilla | 0.6699 | - | - |
| Oracle | 0.6812 | - | - |
| WinAdapt | **0.8524** | 0.3198 | 0.2031 |

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
