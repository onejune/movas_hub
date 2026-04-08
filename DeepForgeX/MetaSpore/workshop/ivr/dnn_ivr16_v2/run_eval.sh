#!/bin/bash
source /mnt/workspace/walter.wan/utils/scripts/dnn_lib_common.sh

init_env

cat > ./temp/eval_only.py << 'EVAL_SCRIPT'
import sys
import time
sys.path.insert(0, './src')
from dnn_trainFlow import DNNModelTrainFlow

flow = DNNModelTrainFlow('./conf/widedeep.yaml')
flow._init_spark()

# 手动设置必要的属性
flow.eval_keys = "business_type"
flow.val_start_time = time.time()

flow._run_evaluation_manual(model_date="2026-03-02", sample_date="2026-03-03")
flow._stop_spark()
EVAL_SCRIPT

$PYTHON_ENV ./temp/eval_only.py
