source /mnt/workspace/walter.wan/utils/lib_common.sh

# 参数校验
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_date> <validation_date>"
    exit 1
fi
OSS_SAMPLE_PATH="/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v8/csv"

model_date="$1"
sample_date="$2"

validation_ivr "$model_date" "$sample_date"
