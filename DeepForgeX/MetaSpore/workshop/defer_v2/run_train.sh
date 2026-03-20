#!/bin/bash
# DEFER v2 训练启动脚本
# 
# 使用方式:
#   ./run_train.sh                    # 默认训练
#   ./run_train.sh --validation       # 仅验证
#
set -e

cd "$(dirname "$0")"

# ========== 配置 ==========
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
METASPORE_DIR="/mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/MetaSpore/python"
WD_V5_DIR="/mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/MetaSpore/workshop/wd_v5"
LOG_DIR="./log"

mkdir -p ./output ${LOG_DIR}

export PYTHONPATH="${METASPORE_DIR}:${WD_V5_DIR}:$PYTHONPATH"
export PYSPARK_PYTHON=${PYTHON_ENV}
export PYSPARK_DRIVER_PYTHON=${PYTHON_ENV}

# 复制依赖
cp ${WD_V5_DIR}/metrics_eval.py ${WD_V5_DIR}/movas_logger.py ${WD_V5_DIR}/feishu_notifier.py ./

# 打包 python.zip (Spark executor 需要)
echo "打包 python.zip..."
cd ${METASPORE_DIR}/..
rm -f python.zip
zip -rq python.zip python -x "*.pyc" -x "__pycache__/*"
mv python.zip $(dirname "$0")/
cd - > /dev/null

# ========== 运行 ==========
CONF="${1:-conf/config.yaml}"
LOG_FILE="${LOG_DIR}/log.log"

echo "Config: ${CONF}"
echo "Log: ${LOG_FILE}"

nohup ${PYTHON_ENV} ms_defer.py --conf "${CONF}" --eval_keys business_type 2>&1 | tee ${LOG_FILE} &

echo "PID: $!"
echo "tail -f ${LOG_FILE}"
