# Defer PyTorch 版本记录

## 版本对比

| 版本 | AUC (MA) | PR-AUC | LogLoss | ECE | PCOC | 说明 |
|------|----------|--------|---------|-----|------|------|
| Oracle | 0.8042 | 0.2466 | 0.2236 | 0.0127 | 1.03 | 理论上界 |
| **v1 基础修复版** | **0.8035** | 0.2462 | 0.2247 | 0.0147 | 1.09 | 修复 AUC=0.5 bug |
| v2 TF loss 对齐 | 0.8034 | 0.2461 | 0.2247 | 0.0147 | 1.05 | 对齐 TF 原版 loss |
| v3 全面对齐 | 0.7720 | 0.2087 | 0.2456 | 0.0326 | 0.99 | per-sample time_win + 三类样本 |
| FNC | 0.7025 | 0.1822 | 0.2759 | 0.0520 | 1.40 | 基线 |
| ES-DFM (iw) | 0.6780 | 0.1777 | 0.2982 | 0.0470 | 0.63 | 重要性加权版 |

---

## v1：基础修复版（最优）

**代码位置**：`src_pytorch/`（git tag: `v3_full_align` 之前的状态，见下方关键差异）

**修复内容**：
- 修复 `--save_path` 参数不支持
- 修复多输出模型评估用错 logits 维度（改用 `cv_logits`）
- 修复 `delay_win_select_loss` 标签格式
- 修复 ES-DFM 模型输出 `cv_logits = -tn_logits`
- 添加 `add_oracle_labels()` 方法

**关键实现**：
- `add_winadapt_labels(cut_sec, win1, win2, win3)`：不做样本扩展，所有样本统一用 `win3` 作为观察窗口
- `delay_win_select_loss`：简化版，负样本 = `1 - label`
- BN 位置：`Linear → BN → leaky_relu`

**结果**：AUC=0.8035，接近 Oracle（0.8042），差距仅 0.0007

---

## v2：TF loss 对齐版

**代码位置**：`versions/v2_tf_loss/`（data.py + metrics.py，其余同 v1）

**改动内容**（相对 v1）：
- `loss.py`：重写 `delay_win_select_loss`，对齐 TF 原版三套并行 loss：
  - `loss_win_15/30/60`（stop_grad 版，权重 0.10/0.05/0.05）
  - `loss_win_15/30/60_spm`（精确负样本版，权重 0.10/0.05/0.05）
  - `loss_cv_spm`（延迟正样本也监督 cv_logits，权重 0.60）
- 精确负样本：`label_15_0 = label_11_15 + label_11_30 + label_11_60 + label_10`

**结果**：AUC=0.8034（与 v1 持平，loss 对齐影响不大）

---

## v3：全面对齐 TF 原版

**代码位置**：`versions/v3_full_align/`（完整四个文件）

**改动内容**（相对 v2）：
- `data.py`：`add_winadapt_labels` 重写
  - per-sample `time_win`：按 `bt_max_delay` 配置，每个 business_type 独立观察窗口
  - 三类样本混合：通常样本 + 延迟正样本(label_11) + 真负样本(label_10)
  - `label_01_30/60_mask` 基于 `time_win` 动态计算
  - `get_stream_data` 返回三元组，增加 `test_stream_nowin`
- `models.py`：BN 位置对齐 TF 原版，改为 `Linear → leaky_relu → BN`
- `train.py`：支持 nowin 评估，输出 nowin AUC/PR-AUC/LogLoss/ECE

**结果**：AUC=0.7720（低于 v1/v2）

**分析**：
- per-sample time_win 使短窗口业务（bt 最大延迟 2~24h）的训练信号减少
- BN 位置改动对预训练权重不友好，收敛变慢
- PCOC 从 1.09→0.99，校准变好，但排序能力下降

---

## 文件结构

```
versions/
├── README.md          # 本文件
├── v2_tf_loss/
│   ├── data.py        # 同 v1（未改动）
│   └── metrics.py     # 同 v1（未改动）
│   # loss.py 见 src_pytorch/（v2 版本已是当前 loss.py）
└── v3_full_align/
    ├── data.py        # per-sample time_win + 三类样本
    ├── loss.py        # 同 v2（TF loss 对齐）
    ├── metrics.py     # 同 v1
    ├── models.py      # BN 位置对齐 TF 原版
    └── train.py       # 支持 nowin 评估
```

> 注：v1 代码即为 git 首次提交时的 `src_pytorch/`，可通过 `git show v3_full_align:src_pytorch/` 查看 v3 当前状态。
> v1 的 loss.py 未单独保存，关键差异见上方说明。

---

## 运行命令

```bash
# v1/v2/v3 通用命令
python3 src_pytorch/train.py \
    --method delay_win_adapt --mode stream \
    --data_path ./data/business_64d_v3.txt \
    --batch_size 2048 --epoch 1 --lr 0.001 --l2_reg 1e-6 \
    --C 24 --win1 24 --win2 48 --win3 72 \
    --stream_start 30 --stream_mid 30 --stream_end 60 \
    --seed 42
```
