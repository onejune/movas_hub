# dnn_ivr16_v2 - 保守特征精简实验

## 实验目标
在 baseline (dnn_ivr16_v1) 基础上进行保守的特征精简，目标是 **AUC 下降 < 0.1%**。

## 特征变化
- **Baseline**: 340 特征
- **v2**: 253 特征 (减少 25.6%)

### 删除的特征 (87 个)
1. **长时间窗口** (60d/90d/180d) - 信息衰减严重，与短期特征高度相关
2. **bucket 特征** (9 个) - 与原始数值特征重复
3. **部分冗余时间窗口** - 如 31_60d, 61_90d, 61_180d 等

### 新增特征 (6 个交叉特征)
- country#demand_pkgname
- business_type#devicetype
- adx#bundle
- country#business_type
- os#demand_pkgname
- country#adx

### 保留的核心特征
- ✅ 所有上下文特征 (country, city, adx, bundle, devicetype, os 等)
- ✅ 所有 ID 特征 (campaignid, offerid, adid, video_id, rta_id 等)
- ✅ 实时特征 (ruf2_*, huf_*)
- ✅ 1d/3d/7d/15d/30d 时间窗口

## 对比 slim 版本
| 版本 | 特征数 | AUC | 变化 |
|------|--------|-----|------|
| baseline | 340 | 0.8427 | - |
| slim | 179 | 0.8361 | -0.66% ❌ |
| v2 | 253 | ? | 目标 < -0.1% |

## 运行
```bash
cd /mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/workshop/ctr/dnn_ivr16_v2
bash run_train.sh
```
