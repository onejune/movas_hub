# SID 生成 Pipeline

一键式完成 商品文本 → Embedding → RQ-VAE训练 → SID 的全流程。

## 快速开始

```bash
cd /mnt/workspace/MiniOneRec/rq/pipeline

# 1. 编辑配置（设置GPU/CPU、模型路径等）
vim config.sh

# 2. 安装环境（只需执行一次）
bash step1_setup.sh

# 3. 下载模型（只需执行一次）
bash step2_download_model.sh

# 4. 一键运行全流程
bash run_all.sh

# 5. 生成汇总表格和质量评估
bash step7_generate_summary.sh
bash step8_evaluate_quality.sh
```

## 文件结构

```
pipeline/
├── config.sh                    # 【核心】统一配置文件，改这一个就够
├── step1_setup.sh               # 安装环境和依赖
├── step2_download_model.sh      # 下载 Qwen 模型
├── step3_text2emb.sh            # 文本 → Embedding
├── step4_train_rqvae.sh         # 训练 RQ-VAE
├── step5_generate_sid.sh        # Embedding → SID
├── step6_text2sid.sh            # 端到端：文本 → SID（快捷方式）
├── step7_generate_summary.sh    # 生成汇总表格（文本+Embedding+SID）
├── step8_evaluate_quality.sh    # SID 质量评估
├── run_all.sh                   # 一键全流程（Step3→4→5）
├── README.md                    # 本文件
├── scripts/                     # Python 脚本（不需要直接执行）
│   ├── text2emb.py              # 文本转embedding
│   ├── train_rqvae.py           # 训练RQ-VAE
│   ├── generate_sid.py          # 生成SID
│   ├── text2sid.py              # 端到端脚本
│   ├── generate_summary.py      # 生成汇总表格
│   ├── evaluate_sid_quality.py  # SID质量评估
│   ├── datasets.py              # 数据集加载
│   ├── trainer.py               # 训练器
│   ├── utils.py                 # 工具函数
│   └── models/                  # RQ-VAE 模型
│       ├── __init__.py
│       ├── rqvae.py
│       ├── rq.py
│       ├── vq.py
│       └── layers.py
└── output/                      # 【所有输出都在这里】
    ├── embeddings/              # Step3 输出: .npy embedding文件
    ├── checkpoints/             # Step4 输出: RQ-VAE .pth模型文件
    ├── sids/                    # Step5 输出: .json SID文件
    ├── summary/                 # Step7 输出: 汇总表格 .csv/.json
    └── evaluation/              # Step8 输出: 质量评估报告 .json
```

## GPU/CPU 切换

打开 `config.sh`，修改设备配置即可：

```bash
# ======================== 设备配置 ========================
DEVICE_TEXT2EMB="cpu"          # Step3: 文本转Embedding
DEVICE_TRAIN_RQVAE="cuda:0"   # Step4: 训练RQ-VAE（推荐GPU）
DEVICE_GENERATE_SID="cpu"     # Step5: 生成SID
DEVICE_TEXT2SID="cpu"          # Step6: 端到端
```

**建议**：
- Step3（文本转Embedding）：7B模型需要约15GB显存，没有大显存GPU就用 `cpu`
- Step4（训练RQ-VAE）：**强烈推荐GPU**，CPU要跑几十个小时，GPU只需几小时
- Step5（生成SID）：很快，`cpu` 就够
- Step6（端到端）：取决于模型大小

## 输出结果说明

| 步骤 | 输出位置 | 文件格式 | 说明 |
|------|----------|----------|------|
| Step3 | `output/embeddings/*.npy` | numpy数组 | 每个商品一行向量（N × D），如 10000×3584 |
| Step4 | `output/checkpoints/*/best_collision_model.pth` | PyTorch模型 | 训练好的RQ-VAE，包含编码器+量化器+解码器 |
| Step5 | `output/sids/*.index.json` | JSON | 最终SID，格式：`{"0": ["<a_42>","<b_80>","<c_160>"], ...}` |
| Step6 | `output/sids/*.text2sid.json` | JSON | 端到端生成的SID，同上格式 |
| Step7 | `output/summary/*.csv` / `.json` | CSV/JSON | 汇总表格：文本+Embedding+SID对照 |
| Step8 | `output/evaluation/*_quality.json` | JSON | SID质量评估报告（碰撞率、语义分组准确率等） |

### 输出示例

**Embedding** (`output/embeddings/Toys_and_Games_10k.emb-qwen-td.npy`):
```
shape: (10000, 3584)   # 10000个商品，每个3584维向量
```

**SID** (`output/sids/Toys_and_Games_10k.index-qwen.json`):
```json
{
  "0": ["<a_192>", "<b_195>", "<c_84>"],
  "1": ["<a_42>", "<b_80>", "<c_160>"],
  "2": ["<a_88>", "<b_156>", "<c_42>"],
  ...
}
```

## 分步执行说明

### Step 1: 安装环境
```bash
bash step1_setup.sh
conda activate minionerec-sid
```

### Step 2: 下载模型
```bash
# 修改 config.sh 中的 QWEN_MODEL_PATH 选择想要的模型
bash step2_download_model.sh
```

### Step 3: 文本 → Embedding
```bash
bash step3_text2emb.sh
# 输出: output/embeddings/Toys_and_Games_10k.emb-qwen-td.npy
```

### Step 4: 训练 RQ-VAE
```bash
bash step4_train_rqvae.sh
# 输出: output/checkpoints/Toys_and_Games_10k/{时间戳}/best_collision_model.pth

# ⚠️ 训练完成后，将模型路径填入 config.sh:
# RQVAE_CKPT_PATH="output/checkpoints/Toys_and_Games_10k/{时间戳}/best_collision_model.pth"
```

### Step 5: 生成 SID
```bash
bash step5_generate_sid.sh
# 输出: output/sids/Toys_and_Games_10k.index-qwen.json
```

### Step 6: 端到端（可选快捷方式）
```bash
# 编辑 step6_text2sid.sh 选择模式（单条/批量/交互），然后：
bash step6_text2sid.sh
```

### Step 7: 生成汇总表格
```bash
bash step7_generate_summary.sh
# 输出: 
#   output/summary/Toys_and_Games_10k_summary.csv  (Excel可直接打开)
#   output/summary/Toys_and_Games_10k_summary.json (包含完整embedding)
```

**汇总表格内容**：
- 商品ID、标题、文本
- Embedding 统计信息（维度、范数、均值）+ 前N维数值
- SID（完整3层 + 分层显示）

### Step 8: SID 质量评估
```bash
bash step8_evaluate_quality.sh
# 输出: output/evaluation/Toys_and_Games_10k_quality.json
```

**评估指标**：
- **碰撞率**：有多少商品的 SID 完全相同
- **语义分组准确率**：embedding 最相似的 K 个邻居中，有多少在同一个 SID 组
  - 前1层准确率（粗粒度分组）
  - 前2层准确率（中粒度分组）
  - 前3层准确率（细粒度分组）
- **随机基线 & 提升倍数**：对比随机分配的效果

**示例输出**：
```
======================================================================
SID 质量评估总结
======================================================================
  商品数:        10000
  SID层数:       3
  碰撞率:        0.05%

  ★ 语义分组准确率 (K=10):
     前1层: 45.2%  (随机基线 0.5%, 提升 90.4x, 256个组)
     前2层: 28.1%  (随机基线 0.04%, 提升 702.5x, 65536个组)
     前3层: 15.3%  (随机基线 0.0002%, 提升 76500x, 16777216个组)
======================================================================
```

## 更换模型

1. 修改 `config.sh` 中的 `QWEN_MODEL_PATH`
2. 重新执行 `step2_download_model.sh`
3. 从 Step3 开始重新跑（因为 embedding 维度会变）
