# CTR Auto HyperOpt 开发记录索引

CTR 预估自动化模型选择与超参优化框架。

每天的开发记录和评估结果见 `docs/changelogs/` 目录：

| 日期 | 主要内容 |
|------|---------|
| [2026-03-15](changelogs/2026-03-15.md) | 项目初始化、FLAML ML 搜索、DeepCTR/MLGB 集成、Optuna 超参寻优 |

---

## 最新评估结果（持续更新）

### 数据集
- 路径：`/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/`
- 64 天分区 parquet，单日约 24 万样本
- 转化率：7.4%
- 特征：9 个类别特征 + 4 个交叉特征

### 模型对比结果（2026-03-15，10% 采样）

| 排名 | 模型 | 类型 | AUC | 备注 |
|------|------|------|-----|------|
| 🥇 1 | **xDeepFM** | DL | **0.7999** | DeepCTR, cin_layer_size=(128,128) |
| 🥈 2 | AutoInt | DL | 0.7985 | DeepCTR |
| 🥉 3 | FiBiNet | DL | 0.7971 | DeepCTR |
| 4 | MLP (Optuna) | DL | 0.7968 | 自定义 PyTorch |
| 5 | XGBoost | ML | 0.7967 | FLAML |
| 6 | Wide & Deep (Optuna) | DL | 0.7934 | 自定义 PyTorch |
| 7 | LightGBM | ML | 0.7932 | FLAML |
| 8 | CatBoost | ML | 0.7922 | FLAML |

> xDeepFM 最优，CIN 结构对特征交叉效果显著
