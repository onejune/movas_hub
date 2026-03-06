#!/bin/bash

#cp ../alphaFM-master/bin/fm* ../model_bin/


# 设置起始日期和结束日期
start_date="20250220"
end_date="20250323"
#val_date="20250204"

# 将日期转换为可以递增的格式
start_date=$(date -d "$start_date" +"%Y%m%d")
end_date=$(date -d "$end_date" +"%Y%m%d")

# 初始化当前日期为起始日期
current_date=$start_date

# 进入循环，直到当前日期大于结束日期
while [ "$current_date" -le "$end_date" ]; do
    echo "=================== train date: $current_date ===================="
    train_date=$(date -d "$current_date" +"%Y-%m-%d")
    time sh train_fm.sh $train_date
    # 计算下一天的日期
    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done

val_date=$(date -d "$val_date" +"%Y-%m-%d")
echo "=================== validation date: $val_date ==================="

#time sh val_by_model.sh $train_date $val_date

time sh val_by_model.sh 20250322 20250323
