#!/bin/bash

# 设置起始日期和结束日期
start_date=$1
end_date=$2

# 将日期转换为可以递增的格式
start_date=$(date -d "$start_date" +"%Y%m%d")
end_date=$(date -d "$end_date" +"%Y%m%d")

# 初始化当前日期为起始日期
current_date=$start_date

# 进入循环，直到当前日期大于结束日期
while [ "$current_date" -le "$end_date" ]; do
    echo "=================== train date: $current_date ===================="
    train_date=$(date -d "$current_date" +"%Y-%m-%d")
    sh train_one_day.sh $train_date
    # 计算下一天的日期
    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done

model_output="./train_output/base"
#sort -k2,2 -t$'\002' -r -g $model_output > ${model_output}.sort
awk -F'\002' '{if($2!=0) print $0}' $model_output | sort -k2,2 -t$'\002' -r -g > ${model_output}.filt
wc -l ${model_output}.filt
pwd
