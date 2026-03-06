source './func.sh'
root_data_path="oss://spark-ml-train-new/liufashuai/dsp/cps/purchase_model/7_train_data"
model_output=train_output/base
log_file=log/log.$cur_date
business='purchase'

set -x
echo `pwd`

function prepare_model(){
    return 0    
}

function prepare_data1(){
    data_day=$1
    data_path=$2
    log "INFO" "开始准备数据, data_day:$data_day, data_path:$data_path"
    if [[ -f "$data_path" ]]; then
	    log "INFO" "$data_path已经存在,直接训练!"
    else
        # 下载训练数据
        rm -rf ./temp/*
        ossutil64 cp ${root_data_path}/${data_day}/ ./temp/ --recursive
        find ./temp/ -type f -name "*.csv" -exec cat {} >> ${data_path} \;
        if [ $? -ne 0 ]; then
            echo "merge data error!"
            exit 255
        fi
    fi
}

function prepare_data2(){
    data_day=$1
    data_path=$2
    log "INFO" "开始准备数据, last_day:$data_day"
    # 下载训练数据
    local oss_file_path=`ossutil64 ls ${root_data_path}/${data_day}/ --config-file ./ossutilconfig | grep part | tail -1`
    if [ -z "${oss_file_path}" ]; then
        echo "no data file!"
        exit 255
    fi

    ossutil64 cp -f ${oss_file_path} ${data_path} --config-file ./ossutilconfig
    if [ $? -ne 0 ]; then
        echo "download train data error!"
        exit 255
    fi
}

function train_model()
{
    local jar=$1
    local output=$2
    local data=$3
    local conf=$4
    rm -rf $output/auc
    java -jar -Xmx80g $jar -i $data -c $conf -f f
    if [ $? -ne 0 ]; then
        echo "train error!"
        exit 255
    fi
}

function prepare_4_train() {
    local train_data=$1
    local base_jar="./ftrl_maven_walter-0.0.1-SNAPSHOT.jar"
    
    train_model ${base_jar} "train_output/" $train_data "conf/"
    echo "train completely....."

    #sort -k2,2 -t$'\002' -r -g $model_output > ${model_output}.sort
    cp $model_output ${model_output}.${train_date}
}

function upload_model() {
    return 0
}

cur_time=`date -d  "0 day ago " +%Y%m%d%H`
cur_date=`date -d  "0 day ago " +%Y%m%d`
if [ -z "$1" ]; then
    last_day=`date  -d "1 day ago" +%Y%m%d`
    train_date=$last_day
else
    last_day=$1
    #train_date: 2024-09-01
    train_date=$1
fi

echo "train_date:$train_date, cur_time:$cur_time"
data_path=../sample/sample_${train_date}

prepare_model
echo "log_file: "$log_file

prepare_4_train ${data_path}
if [ $? -ne 0 ]; then
    echo "train error!"
    exit 255
fi

