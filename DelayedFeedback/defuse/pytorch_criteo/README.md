# DEFUSE PyTorch Implementation - Criteo Benchmark

延迟反馈转化率预测方法对比实验（Criteo 数据集）

## 快速开始

```bash
# 运行单个方法
./run.sh DEFER

# 运行所有方法
./run.sh all

# 查看结果
./run.sh results
```

## 实验结果 (2026-03-22)

| Rank | Method | AUC | PR-AUC | LogLoss | Time |
|------|--------|-----|--------|---------|------|
| 1 | **DEFER** | **0.8429** | 0.5874 | 0.4390 | 27.7m |
| 2 | ES-DFM | 0.8417 | 0.5899 | 0.3521 | 22.7m |
| 3 | DEFUSE | 0.8416 | 0.5873 | 0.4064 | 22.7m |
| 4 | Oracle | 0.8413 | 0.5950 | 0.3494 | 13.2m |
| 5 | Bi-DEFUSE | 0.8411 | 0.5941 | 0.3498 | 14.7m |
| 6 | Vanilla | 0.8239 | 0.5692 | 0.4072 | 184.8m |
| 7 | FNW | 0.8216 | 0.5572 | 0.5333 | 190.3m |
| 8 | FNC | 0.8216 | 0.5454 | 0.6219 | 187.2m |

## 结论

1. **DEFER 效果最好**（AUC=0.8429），超过 Oracle baseline
2. ES-DFM / DEFUSE / Bi-DEFUSE 效果相近，接近 Oracle
3. FNW/FNC 比 Vanilla 还差，简单加权策略无效

## 方法说明

| Method | 策略 | 需要预训练 |
|--------|------|-----------|
| Vanilla | 延迟转化视为负样本 | ❌ |
| FNW | Fake Negative Weighted | ❌ |
| FNC | Fake Negative Calibration | ❌ |
| DFM | Delayed Feedback Model | ❌ |
| **DEFER** | Duplicate samples for delayed positives | ✅ |
| **ES-DFM** | Elapsed-time Sampling DFM | ✅ |
| **DEFUSE** | In-window / Out-window dual prediction | ✅ |
| Bi-DEFUSE | Bidirectional DEFUSE | ❌ |
| Oracle | 使用真实标签（upper bound） | ❌ |

## 训练流程

### 1. 不需要预训练的方法 (Vanilla, FNW, FNC, DFM, Oracle, Bi-DEFUSE)

```
数据加载 → 训练 1 epoch → 评估
```

### 2. 需要预训练的方法 (DEFER, ES-DFM, DEFUSE)

```
数据加载 → 预训练 tn/dp 模型 → 主模型训练 → 评估
           (预测真负/延迟正概率)   (使用 tn/dp 概率做重要性加权)
```

**预训练目的**：
- 训练辅助模型预测 `tn_prob`（真负样本概率）和 `dp_prob`（延迟正样本概率）
- 主训练时用这些概率校正延迟反馈带来的标签噪声

## 数据集

- **Criteo Delayed Feedback Dataset**
- 路径: `/mnt/workspace/walter.wan/open_research/criteo_dataset/data.txt`
- 样本数: 15,898,883
- 正样本率: 18.51%（oracle labels）
- 观察窗口: 1 小时
- 归因窗口: 7 天

## 文件结构

```
pytorch_criteo/
├── run.sh              # 🚀 入口脚本
├── README.md           # 📖 说明文档
├── config.py           # ⚙️ 配置参数
├── data.py             # 📊 数据处理
├── models.py           # 🧠 模型定义
├── loss.py             # 📉 损失函数
├── metrics.py          # 📈 评估指标
├── trainer.py          # 🏃 训练器
├── run_full_parallel.py    # 主运行脚本
├── run_quick_test.py       # 快速测试（1%数据）
├── results_full/       # 实验结果
└── logs_full/          # 训练日志
```

## 核心代码说明

### config.py
- 数据路径、时间窗口配置
- 模型超参数（hidden_dims, embed_dim）
- 训练参数（batch_size, lr, epochs）

### data.py
- `DataDF`: 数据容器类
- `form_vanilla()`: Vanilla 数据处理
- `form_oracle()`: Oracle 数据处理
- `add_defer_duplicate_samples()`: DEFER 数据增强
- `construct_tn_dp_data()`: 构建预训练数据

### models.py
- `BaseMLP`: 基础 MLP 模型
- `MLP_SIG`: Sigmoid 输出模型
- `MLP_tn_dp`: 预训练模型（输出 tn/dp logits）
- `BiDefuseModel`: Bi-DEFUSE 模型

### loss.py
- `vanilla_loss`: BCE loss
- `fnw_loss`: Fake Negative Weighted loss
- `defer_loss`: DEFER importance weighted loss
- `esdfm_loss`: ES-DFM loss
- `defuse_loss`: DEFUSE loss

## 环境要求

```
Python 3.9+
PyTorch 2.0+
pandas, numpy, tqdm, scikit-learn
```

CPU-only 训练，无需 GPU。

---
*最后更新: 2026-03-22*
