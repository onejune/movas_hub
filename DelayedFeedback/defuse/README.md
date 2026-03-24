# DEFUSE - Delayed Feedback Modeling

延迟反馈转化率预测方法对比实验

## 目录结构

```
DEFUSE/
├── pytorch_criteo/     # 🔥 PyTorch 实现 (推荐，CPU 可跑)
├── pytorch_defer/      # PyTorch 实现 (defer 私有数据)
└── tf_criteo/          # TensorFlow 原始实现 (需要 GPU)
```

## 快速开始

```bash
cd pytorch_criteo
./run.sh DEFER          # 运行单个方法
./run.sh results        # 查看结果
```

## 实验结果 (Criteo Dataset)

| Rank | Method | AUC |
|------|--------|-----|
| 1 | **DEFER** | **0.8429** |
| 2 | ES-DFM | 0.8417 |
| 3 | DEFUSE | 0.8416 |
| 4 | Oracle | 0.8413 |
| 5 | Bi-DEFUSE | 0.8411 |
| 6 | Vanilla | 0.8239 |
| 7 | FNW | 0.8216 |
| 8 | FNC | 0.8216 |

## 方法对比

| Method | 核心思想 |
|--------|---------|
| Vanilla | 延迟转化视为负样本（baseline） |
| FNW/FNC | 简单加权/校准 |
| DFM | 延迟反馈建模 |
| DEFER | 延迟正样本复制增强 |
| ES-DFM | Elapsed-time 重要性采样 |
| DEFUSE | In-window/Out-window 双预测 |
| Oracle | 使用真实标签（upper bound） |

## 参考论文

- DEFUSE: Delayed Feedback Modeling with Unbiased Estimation
- ES-DFM: Elapsed-Time Sampling for Delayed Feedback Modeling

---
*详细说明见各子目录 README*
