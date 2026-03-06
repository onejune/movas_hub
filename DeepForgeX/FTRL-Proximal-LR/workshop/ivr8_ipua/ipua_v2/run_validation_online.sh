source /mnt/workspace/walter.wan/utils/lib_common.sh

OSS_SAMPLE_PATH="/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v4/csv"

model_date="$1"
sample_date="$2"
validation_online_model "$model_date" "$sample_date"
