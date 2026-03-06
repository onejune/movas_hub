source /mnt/workspace/walter.wan/utils/lib_common.sh

# 参数校验
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_date> <validation_date>"
    exit 1
fi

model_date="$1"
sample_date="$2"
model_dir="/mnt/data/oss_dsp_algo/ivr/model/ruf_v7"

validation_online_model "$model_dir" "$model_date" "$sample_date"
