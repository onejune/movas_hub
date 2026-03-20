# MetaSpore Workshop

实验目录，按模型类型分类组织。

## 目录结构

```
workshop/
├── ctr/        # CTR/CVR 单任务模型
│   ├── dcn_v*      # Deep & Cross Network
│   ├── dfm_v*      # DeepFM
│   └── wd_v*       # Wide & Deep
│
├── defer/      # 延迟反馈模型 (DEFER)
│   ├── defer_v1    # 短窗口 15/30/60min
│   └── defer_v2    # 长窗口 24/48/72h
│
├── ltv/        # LTV 预估模型
│   ├── ltv_mse_*       # MSE Loss
│   ├── ltv_quantile_*  # Quantile Regression
│   ├── ltv_star_*      # STAR 架构
│   ├── ltv_wce_*       # Weighted Cross Entropy
│   └── ltv_ziln_*      # Zero-Inflated Log Normal
│
├── mtl/        # 多任务学习 (Multi-Task Learning)
│   ├── mtl_home_*      # Home 场景
│   ├── mtl_mmoe_*      # MMoE 架构
│   ├── mtl_pepnet*     # PEPNet 架构
│   ├── mtl_ple_*       # PLE 架构
│   ├── mtl_sa_mmoe_*   # Self-Attention MMoE
│   ├── mtl_sbm_*       # SBM 架构
│   └── mtl_uw_mmoe_*   # Uncertainty Weighted MMoE
│
├── mdl/        # 多域学习 (Multi-Domain Learning)
│   ├── mdl_ada_sparse_* # Adaptive Sparse
│   ├── mdl_dswin_*      # Domain-Specific Window
│   ├── mdl_hmoe_*       # Hierarchical MoE
│   ├── mdl_mmoe_*       # Multi-Domain MMoE
│   ├── mdl_pepnet*      # Multi-Domain PEPNet
│   ├── mdl_sacn_*       # SACN 架构
│   ├── mdl_sain_*       # SAIN 架构
│   ├── mdl_star*        # STAR 架构
│   └── mdl_starGate_*   # StarGate 架构
│
├── winrate/    # 竞价/出价模型
│   ├── winrate_deepcensor_* # DeepCensor
│   └── winrate_sim_*        # Simulation
│
└── archive/    # 归档 (旧版本/实验性)
    ├── wd_diff_comb_v1     # 特征组合实验
    ├── wd_sample_exp_v1    # 采样实验
    ├── mtl_mmoe_v5_*       # MMoE v5 子版本
    ├── mtl_pepnet2_v4/v5   # PEPNet 实验版本
    └── gr/                 # 其他
```

## 命名规范

- `{模型类型}_{架构}_{版本}`: 例如 `mtl_mmoe_v5`
- 版本号递增表示配置/数据/特征变化
- 子版本 `_2`, `_3` 表示小改动，建议归档

## 常用实验

| 任务 | 推荐目录 | 说明 |
|------|----------|------|
| CTR 预估 | `ctr/wd_v5` | Wide & Deep 基线 |
| 延迟反馈 | `defer/defer_v2` | 24/48/72h 窗口 |
| 多任务学习 | `mtl/mtl_mmoe_v8` | 最新 MMoE |
| 多域学习 | `mdl/mdl_star_v9` | 最新 STAR |
| LTV 预估 | `ltv/ltv_ziln_v2` | ZILN 模型 |
