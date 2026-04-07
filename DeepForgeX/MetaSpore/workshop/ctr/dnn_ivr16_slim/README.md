# dnn_ivr16_slim - 特征精简实验

## 实验目标
1. **特征精简**: 删除冗余时间窗口特征，减少模型复杂度
2. **交叉特征**: 新增6个高价值交叉特征，提升模型表达能力

## 特征变化

### 原始特征 (dnn_ivr16_v1): 341个
### 精简后特征 (v2_slim): 179个

### 删除的特征类型

| 类别 | 删除原因 | 删除数量 |
|------|---------|---------|
| bucket特征 | 与原始特征冗余 | ~27个 |
| 冗余时间窗口 | 保留3个最优窗口(如3d/7d/30d) | ~100个 |
| 长时间窗口 | 60d/90d/180d信息衰减严重 | ~30个 |
| 低区分度特征 | carrier, display_manager等 | ~5个 |

### 新增交叉特征 (6个)

| 交叉特征 | 业务含义 |
|---------|---------|
| country#demand_pkgname | 地区×应用偏好 |
| business_type#devicetype | 业务线×设备类型 |
| adx#bundle | 流量源×媒体 |
| country#business_type | 地区×业务线 |
| os#demand_pkgname | 系统×应用 |
| country#adx | 地区×流量源 |

### 时间窗口精简策略

| 特征类型 | 原始窗口 | 精简后窗口 |
|---------|---------|-----------|
| 实时特征(RUF) | 3m/10m/30m/1h/3h/12h/24h/48h | 3m/1h/12h/48h |
| 小时特征(HUF) | 1h/3h/12h/24h/48h | 1h/12h/48h |
| 日级特征(DUF) | 1d/3d/7d/15d/30d/60d/90d/180d | 1d/3d/7d/30d |
| 分段窗口 | 1_3d/4_10d/11_30d/31_60d/61_90d/61_180d | 1_3d/4_10d/11_30d |

## 预期效果

1. **训练速度**: 特征数减少47%，预计训练速度提升30-40%
2. **模型大小**: Embedding参数减少，模型更轻量
3. **AUC**: 预计持平或略有提升(交叉特征补偿)
4. **泛化能力**: 删除冗余特征可能提升泛化

## 运行方式

```bash
cd /mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/workshop/ctr/dnn_ivr16_v2_slim
bash run_train.sh
```

## 对比实验

| 实验 | 特征数 | 交叉特征 | 状态 |
|-----|-------|---------|------|
| dnn_ivr16_v1 (baseline) | 341 | 0 | ✅ 已完成 |
| dnn_ivr16_slim | 179 | 6 | 🔄 待运行 |

## Baseline结果 (dnn_ivr16_v1)

| 业务线 | AUC | PCOC |
|--------|-----|------|
| Overall | 0.8427 | 1.0497 |
| shopee_cps | 0.8090 | 1.0544 |
| lazada_rta | 0.8000 | 1.0808 |
| lazada_cps | 0.8169 | 1.0140 |
| aecps | 0.8410 | 0.9423 |
| aedsp | 0.8368 | 0.7244 |
| mobplus | 0.7895 | 1.0335 |
| novabeyond | 0.8617 | 1.4013 |
| aerta | 0.8132 | 0.8406 |
| shein | 0.7677 | 1.0102 |
| fluentleads | 0.9437 | 0.9500 |
