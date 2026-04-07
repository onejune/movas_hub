# MetaSpore TrainFlow 重构方案

## 1. 目标

把 `utils/trainflows` 和 `utils/tools` 整合进 `metaspore` 包，实验入口从 shell 改为纯 Python。

---

## 2. 当前结构

```
DeepForgeX/
├── MetaSpore/python/metaspore/      # 框架核心
│   ├── algos/                       # 模型定义
│   └── ...
├── utils/                           # 游离在外的依赖
│   ├── trainflows/                  # 训练流程
│   │   ├── base_trainFlow.py
│   │   ├── dnn_trainFlow.py
│   │   ├── MTL_trainFlow.py
│   │   ├── MDL_trainFlow.py
│   │   ├── ltv_trainFlow.py
│   │   ├── winrate_trainFlow.py
│   │   ├── DELF_trainFlow.py
│   │   ├── defer_trainFlow.py
│   │   └── dense_feature.py
│   ├── tools/                       # 工具类
│   │   ├── movas_logger.py
│   │   ├── metrics_eval.py
│   │   └── feishu_notifier.py
│   └── scripts/
│       └── dnn_lib_common.sh        # shell 入口
└── workshop/                        # 实验目录
    └── ctr/wd_v5/
        ├── run_train.sh             # shell 入口
        └── conf/widedeep.yaml
```

---

## 3. 重构后结构

```
DeepForgeX/
├── MetaSpore/python/metaspore/
│   ├── algos/                       # 模型定义 (不变)
│   ├── trainflows/                  # 【新增】从 utils 迁移
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseTrainFlow
│   │   ├── dnn.py                   # DNNTrainFlow
│   │   ├── mtl.py                   # MTLTrainFlow
│   │   ├── mdl.py                   # MDLTrainFlow
│   │   ├── ltv.py                   # LTVTrainFlow
│   │   └── winrate.py               # WinrateTrainFlow
│   └── utils/                       # 【新增】从 utils/tools 迁移
│       ├── __init__.py
│       ├── logger.py                # MovasLogger
│       ├── metrics.py               # metrics_eval
│       └── notifier.py              # FeishuNotifier
└── workshop/
    └── ctr/wd_v5/
        ├── train.py                 # 【新】Python 入口
        └── conf/widedeep.yaml
```

---

## 4. 实验入口对比

### Before (shell)

```bash
# run_train.sh
source /path/to/dnn_lib_common.sh
TRAINER_SCRIPT_PATH="./src/dnn_trainFlow.py"
model_train
```

### After (Python)

```python
# train.py
from metaspore.trainflows import DNNTrainFlow

if __name__ == "__main__":
    DNNTrainFlow.main()  # 自动解析命令行参数
```

### 运行方式

```bash
# 训练
python train.py --conf ./conf/widedeep.yaml --name wd_v5 --eval_keys business_type

# 仅验证
python train.py --conf ./conf/widedeep.yaml --name wd_v5 --eval_keys business_type \
    --validation True --model_date 2026-04-06 --sample_date 2026-04-07

# 后台运行
nohup python train.py --conf ./conf/widedeep.yaml --name wd_v5 --eval_keys business_type \
    > log/train.log 2>&1 &
```

---

## 5. 包结构设计

### 5.1 metaspore/trainflows/__init__.py

```python
from .base import BaseTrainFlow
from .dnn import DNNTrainFlow
from .mtl import MTLTrainFlow

__all__ = ["BaseTrainFlow", "DNNTrainFlow", "MTLTrainFlow"]
```

### 5.2 metaspore/utils/__init__.py

```python
from .logger import MovasLogger, how_much_time
from .metrics import compute_auc_pcoc, calculate_logloss, PCOC
from .notifier import FeishuNotifier

__all__ = ["MovasLogger", "how_much_time", "compute_auc_pcoc", 
           "calculate_logloss", "PCOC", "FeishuNotifier"]
```

---

## 6. import 路径变化

| Before | After |
|--------|-------|
| `from base_trainFlow import BaseTrainFlow` | `from metaspore.trainflows import BaseTrainFlow` |
| `from movas_logger import MovasLogger` | `from metaspore.utils import MovasLogger` |
| `from metrics_eval import compute_auc_pcoc` | `from metaspore.utils import compute_auc_pcoc` |
| `from feishu_notifier import FeishuNotifier` | `from metaspore.utils import FeishuNotifier` |

---

## 7. 改动清单

| 操作 | 文件 | 状态 |
|------|------|------|
| **新建** | `metaspore/utils/__init__.py` | ✅ 完成 |
| **新建** | `metaspore/utils/logger.py` | ✅ 完成 |
| **新建** | `metaspore/utils/metrics.py` | ✅ 完成 |
| **新建** | `metaspore/utils/notifier.py` | ✅ 完成 |
| **新建** | `metaspore/trainflows/__init__.py` | ✅ 完成 |
| **新建** | `metaspore/trainflows/base.py` | ✅ 完成 |
| **新建** | `metaspore/trainflows/dnn.py` | ✅ 完成 |
| **新建** | `metaspore/trainflows/mtl.py` | ✅ 完成 |
| **新建** | 各实验目录的 `train.py` | ✅ 部分完成 |
| **测试** | import 测试 | ✅ 完成 |

---

## 8. 收益

| 项目 | Before | After |
|------|--------|-------|
| 文件复制 | 每次 ~10 个 | 0 |
| 入口文件 | shell + python 混合 | 纯 Python |
| import 路径 | 相对路径，依赖复制 | 包路径，标准化 |
| IDE 支持 | 差 (找不到定义) | 好 (正常跳转) |
| 维护成本 | 高 (多处修改) | 低 (统一修改) |

---

## 9. 迁移步骤

### Step 1: 创建 metaspore/utils 包 ✅

- `logger.py` - 从 `utils/tools/movas_logger.py` 迁移
- `metrics.py` - 从 `utils/tools/metrics_eval.py` 迁移
- `notifier.py` - 从 `utils/tools/feishu_notifier.py` 迁移

### Step 2: 创建 metaspore/trainflows 包 ⏳

- `base.py` - 从 `utils/trainflows/base_trainFlow.py` 迁移
- `dnn.py` - 从 `utils/trainflows/dnn_trainFlow.py` 迁移
- `mtl.py` - 从 `utils/trainflows/MTL_trainFlow.py` 迁移

### Step 3: 更新 import 路径

所有 `from xxx import yyy` 改为 `from metaspore.xxx import yyy`

### Step 4: 创建实验入口模板

每个实验目录创建 `train.py`

### Step 5: 测试

- import 测试
- 训练流程测试
- 验证流程测试

---

## 10. 回滚方案

保留原有 `utils/` 目录，不删除。如有问题可直接回退到 shell 入口。
