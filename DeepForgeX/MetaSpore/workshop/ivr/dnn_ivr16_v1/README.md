# CTR 训练项目

## 🎯 项目概述

使用 MetaSpore 包结构的 CTR 预估训练项目。

## 📝 README 文件作用

此 README.md 文件不仅用于说明项目结构和使用方法，还用于记录实验的关键信息，包括：
- **实验目的**: 实验的核心目标和假设
- **实验设计**: 数据集、模型配置、超参数设置
- **实验结果**: AUC、PR-AUC、LogLoss 等关键指标
- **结论总结**: 实验发现和下一步计划

实验完成后，可将 README 内容整理成飞书消息发送给相关人员进行汇报。

## 🏗️ 架构特点

### ✅ 旧架构（已废弃）
- 需要复制多个文件到 `src/` 目录
- 相对导入路径
- 文件管理复杂

### ✅ 新架构（当前使用）
- **零文件复制**: 直接使用 MetaSpore 包
- **标准导入**: `from metaspore.trainflows import DNNTrainFlow`
- **统一管理**: 所有依赖在 MetaSpore 包中
- **单一入口**: 一个脚本处理训练和验证
- **代码简化**: train.py 无需硬编码路径
- **易于维护**: 升级只需更新包结构

## 🚀 使用方式

### 训练
```bash
# 使用 shell 脚本（默认为训练）
./run_train.sh
# 或显式指定
./run_train.sh train

# 或直接运行 Python
python train.py --conf ./conf/widedeep.yaml --name PROJECT_NAME --eval_keys business_type
```

### 验证
```bash
# 使用 shell 脚本
./run_train.sh validate 2026-04-06 2026-04-07 business_type ./conf/widedeep.yaml

# 或直接运行 Python
python train.py --conf ./conf/widedeep.yaml --validation True --name PROJECT_NAME \
    --model_date 2026-04-06 --sample_date 2026-04-07 --eval_keys business_type
```

### 帮助
```bash
./run_train.sh help
```

### 后台运行
```bash
nohup ./run_train.sh > log/train.log 2>&1 &
```

## 📁 目录结构

```
project_name/
├── train.py              # 🆕 新版 Python 入口
├── run_train.sh          # 🆕 统一的 shell 入口（训练+验证）
├── conf/                 # 配置文件
├── output/               # 输出目录
├── log/                  # 日志目录
├── src/                  # 🆕 空目录，不再需要复制文件
└── README.md             # 本文档
```

## 🔧 核心代码

### train.py
```python
#!/usr/bin/env python

import sys
import os

def ensure_metaspore_path():
    """确保 MetaSpore 路径在 PYTHONPATH 中"""
    metaspore_dir = "/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/python"
    if metaspore_dir not in sys.path:
        sys.path.insert(0, metaspore_dir)

ensure_metaspore_path()

def main():
    from metaspore.trainflows import DNNTrainFlow
    
    args = DNNTrainFlow.parse_args()
    trainer = DNNTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args)

if __name__ == "__main__":
    main()
```

## 🎁 优势

1. **零文件复制**: 不再需要复制 `dnn_trainFlow.py` 等文件
2. **标准导入**: 使用 `from metaspore.trainflows import DNNTrainFlow`
3. **代码简化**: train.py 无需硬编码路径，依赖 shell 脚本设置的环境
4. **统一维护**: 所有功能在 MetaSpore 包中统一管理
5. **单一入口**: `run_train.sh` 统一处理训练和验证
6. **IDE友好**: 正确的代码跳转和补全
7. **扩展性强**: 新功能只需更新包结构即可使用
```

## 实验记录

### 实验目的
[在此填写实验的核心目标和假设]

### 实验设计
- **数据集**: [数据集名称和规模]
- **模型配置**: [模型类型和参数]
- **超参数**: [学习率、批次大小等]

### 实验结果
- **AUC**: [数值]
- **PR-AUC**: [数值]
- **LogLoss**: [数值]

### 结论总结
[在此填写实验发现和下一步计划]

