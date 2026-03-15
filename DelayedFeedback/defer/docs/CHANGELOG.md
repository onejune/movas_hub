# Defer 开发记录

延迟转化（Delayed Feedback）CVR 预测项目，基于 [DEFER 论文](https://arxiv.org/abs/2104.03208) 实现。

---

## 评估结果汇总

### 数据集
- 文件：`business_64d_v3.txt`，约 2200 万行，70 天
- 转化率：7.5%，最大延迟：7 天
- 特征：9 个类别特征（原始数据第 8-16 列），embedding_dim=8
- 预训练：day 0-30；流式测试：day 30-60（720 个小时窗口）
- 窗口配置：C=24h, win1=24h, win2=48h, win3=72h

### PyTorch 版本结果（2026-03-15）

| 方法 | AUC | PR-AUC | LogLoss | ECE | PCOC | 备注 |
|------|-----|--------|---------|-----|------|------|
| Oracle | 0.8042 | 0.2466 | 0.2236 | 0.0127 | 1.03 | 上界 |
| WinAdapt v1 | **0.8035** | 0.2462 | 0.2247 | 0.0147 | 1.09 | 基础修复版，最优 |
| WinAdapt v2 | 0.8034 | 0.2461 | 0.2247 | 0.0147 | 1.05 | 对齐 TF loss |
| WinAdapt v3 | 0.7720 | 0.2087 | 0.2456 | 0.0326 | 0.99 | 全面对齐 TF 原版 |
| FNC | 0.7025 | 0.1822 | 0.2759 | 0.0520 | 1.40 | |
| ES-DFM (iw) | 0.6780 | 0.1777 | 0.2982 | 0.0470 | 0.6268 | 原版重要性加权 |

> WinAdapt v1 ≈ Oracle（AUC 差距仅 0.0007）

### TF 原版结果（2026-03-15，进行中）

| 方法 | AUC | PR-AUC | LogLoss | 状态 |
|------|-----|--------|---------|------|
| Oracle | 0.8046 | 0.2404 | 0.2186 | ✅ |
| FNC | 0.7183 | 0.1755 | 0.6132 | ✅ |
| Vanilla | 0.6831 | 0.1703 | 0.2882 | ✅ |
| ES-DFM | — | — | — | 🔄 重跑中 |
| DFM | 0.5278 | 0.0851 | 0.8272 | ✅（方法本身局限） |
| WinAdapt | — | — | — | 🔄 重跑中 |

---

## 变更记录

### 2026-03-15

#### 问题诊断与修复（PyTorch）
- **根本原因**：`--save_path` 参数不支持、多输出模型评估使用错误 logits 维度、标签格式与 loss 不匹配，导致 AUC=0.5
- **修复 `src_pytorch/train.py`**：添加 `--save_path` 参数，修复多输出模型（WinAdapt/ES-DFM）评估改用 `cv_logits`，处理 PCOC summary 中的 inf 值
- **修复 `src_pytorch/loss.py`**：重写 `delay_win_select_loss` 和 `delay_tn_dp_loss`，对齐 TF 原版三套并行 loss
- **修复 `src_pytorch/models.py`**：ES-DFM 模型输出 `cv_logits = -tn_logits`
- **修复 `src_pytorch/data.py`**：添加 `add_oracle_labels()` 方法

#### TF 原版 loss 对齐（PyTorch v2）
- 实现三套并行 loss：stop_grad 版 + spm 版 + cv_spm
- 精确负样本：`label_15_0 = label_11_15 + label_11_30 + label_11_60 + label_10`
- 权重：`0.1*win15 + 0.05*win30 + 0.05*win60 + 0.1*win15_spm + 0.05*win30_spm + 0.05*win60_spm + 0.6*cv_spm`

#### PyTorch 三版本存档
- `versions/v1_baseline/`：AUC=0.8035，基础修复版
- `versions/v2_tf_loss/`：AUC=0.8034，对齐 TF loss
- `versions/v3_full_align/`：AUC=0.7720，全面对齐 TF 原版（per-sample time_win）

#### TF 原版修复
- `src_tf/stream_train_test.py`：添加 `_safe_load()` 避免 None ckpt path 崩溃
- `src_tf/stream_train_test.py`：过滤 NaN probs（ES-DFM/WinAdapt 重要性权重可能产生 NaN）
- `src_tf/stream_train_test.py`：auc_ma/ece_ma 跳过 NaN epoch（全负样本窗口）
- `src_tf/data.py`：添加 raw_data pickle cache，避免每次重读 2200 万行
- `src_tf/pretrain.py`：save_weights 前检查 model_ckpt_path 非 None

#### 目录结构整理
- `src/` → `src_tf/`（TF 实现）
- 新增 `scripts/`（所有 run_*.sh）
- 新增 `tools/`（数据处理工具）
- 新增 `docs/CHANGELOG.md`（本文件）

#### business_type 分析
- bt_col_index=0（cate_feat[0]，原始数据 col10）
- 短窗口（≤24h）：bt=[39,48,45,24,49,37,7,56,31,20,62,17,59]，32.3% 样本，CVR=0.0783
- 中窗口（24-72h）：bt=[12,47,53]，18.0% 样本，CVR=0.0035
- 长窗口（>72h）：bt=[60,22,43]，49.4% 样本，CVR=0.0986

---

## 关键结论

1. **WinAdapt ≈ Oracle**：AUC 差距仅 0.0007，延迟建模效果显著
2. **v3 全面对齐 TF 反而 AUC 下降**：per-sample time_win 使短窗口业务训练信号减少，BN 位置改动对预训练权重不友好
3. **ES-DFM 重要性加权效果差**：AUC 仅 0.6780，远低于 WinAdapt
4. **DFM 指数延迟假设太强**：AUC 仅 0.53，不适合业务数据
5. **business_type 最大延迟窗口可作为特征**：不同业务转化模式不同，分层评估更公平

---

## 待办

- [ ] TF 原版 ES-DFM / WinAdapt 重跑完成后更新结果
- [ ] 实现 business_type 分窗口评估逻辑
- [ ] 考虑将最大延迟窗口作为数值/类别特征加入模型
- [ ] 设计公平对比实验（数据新鲜度 trade-off）
