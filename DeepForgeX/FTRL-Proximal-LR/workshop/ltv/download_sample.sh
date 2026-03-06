#!/bin/bash

# 设置起始日期和结束日期
start_date="$1"
end_date="$2"

if [ "$2" = "" ]; then
    end_date=$start_date
fi

# 将日期转换为可以递增的格式
start_date=$(date -d "$start_date" +"%Y%m%d")
end_date=$(date -d "$end_date" +"%Y%m%d")

# 初始化当前日期为起始日期
current_date=$start_date

# 进入循环，直到当前日期大于结束日期
while [ "$current_date" -le "$end_date" ]; do
    echo "=================== train data: $current_date ===================="
    train_date=$(date -d "$current_date" +"%Y-%m-%d")
    rm -rf ./temp/sample_$train_date
    #ossutil cp -f oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_purchase/4_train_data/$train_date/ ./temp/sample_$train_date --recursive
    #ossutil cp -f oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_purchase/5_diff_data_org/$train_date/ ./temp/sample_$train_date --recursive
    #ossutil cp -f oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_purchase/6_noshein_diff_data_org/$train_date/ ./temp/sample_$train_date --recursive
    #ossutil cp -f oss://spark-ml-train-new/liufashuai/dsp/cps/union_purchase/ftrl_lr_purchase/hour_diff_data/$train_date/ ./temp/sample_$train_date --recursive
    ossutil cp -f oss://spark-ml-train-new/wanjun/01_project/ltv/ltv_sample_v1/$train_date/ ./temp/sample_$train_date --recursive

    if [ $? != 0 ]; then
        echo "download train data from oss error: $train_date"
	break
    fi
    if [ ! -n "$(ls "./temp/sample_$train_date")" ]; then
	echo "There is no files in ./temp/sample_$train_date/"
	break
    fi
    mv ./temp/sample_$train_date/part* ./sample/sample_$train_date
    line_count=$(wc -l < ./sample/sample_$train_date)
    if [ "$line_count" -le 1000 ]; then
	echo "line count is invalid for $train_date"
	rm -f ./sample/sample_$train_date
    fi
    echo "line count: $line_count"
    # 计算下一天的日期
    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done

