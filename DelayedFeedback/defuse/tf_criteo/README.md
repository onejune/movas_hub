# DEFUSE TensorFlow Implementation - Criteo Benchmark

原始论文的 TensorFlow 实现（需要 GPU）

## 文件结构

```
tf_criteo/
├── src/                        # TF 源代码
├── delayed_feedback_release/   # 预训练 checkpoints
├── run_pretrain.sh            # 预训练脚本
├── run_base.sh                # 基线方法 (Oracle, Vanilla)
├── run_base_1d.sh             # 1天窗口版本
└── run_defuse.sh              # DEFUSE/Bi-DEFUSE
```

## 运行方式

```bash
# 1. 预训练
./run_pretrain.sh

# 2. 运行基线
./run_base.sh

# 3. 运行 DEFUSE
./run_defuse.sh
```

## 注意事项

- 需要 GPU (CUDA)
- 数据路径需要配置: `--data_path ../data/criteo/data.txt`
- 脚本中大部分命令被注释，取消注释后运行

## 参考

- 论文: DEFUSE: Delayed Feedback Modeling with Unbiased Estimation
- 代码: 原始论文开源实现

---
*PyTorch 实现见 `../pytorch_criteo/`*
