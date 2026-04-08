
## 🚀 使用方式

### 训练
```bash
# 使用 shell 脚本（默认为训练）
./run.sh
# 或显式指定
./run.sh train ./conf/widedeep.yaml

# 或直接运行 Python
python train.py --conf ./conf/widedeep.yaml --name wd_v5 --eval_keys business_type
```

### 验证
```bash
# 使用 shell 脚本
./run.sh validate 2026-04-06 2026-04-07 business_type ./conf/widedeep.yaml

# 或直接运行 Python
python train.py --conf ./conf/widedeep.yaml --validation True --name wd_v5 \
    --model_date 2026-04-06 --sample_date 2026-04-07 --eval_keys business_type
```

### 帮助
```bash
./run.sh help
```

## 📁 目录结构
```
wd_v5/
├── train.py              # 🆕 新版 Python 入口
├── run.sh          # 🆕 统一的 shell 入口（训练+验证）
├── conf/                 # 配置文件
│   └── widedeep.yaml
├── output/               # 输出目录
├── log/                  # 日志目录
└── README.md             # 本文档
```
readme: 用于填写实验描述信息，会在训练结束后通过飞书发送。