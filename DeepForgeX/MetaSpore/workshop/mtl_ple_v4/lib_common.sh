# File 1: lib_common.sh (公共函数库)
#!/bin/bash
set -e

# 配置区域
OSS_SAMPLE_PATH="/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v5/csv"
#OSS_SAMPLE_PATH="oss://spark-ml-train-new/dsp_algo/ivr/sample/ruf_sample_v2/csv"
OSS_ROOT=""
MODEL_OUTPUT_DIR="./train_output"
LOG_DIR="./log"
OSSUTIL_CONFIG="./ossutilconfig"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d%H%M%S).log"
ftrl_jar_path="/mnt/data/oss_dsp_algo/tools/ftrl_jars/ftrl_maven_walter-0.0.1-SNAPSHOT.jar_v1.2"
#ftrl_jar_path="/mnt/workspace/walter.wan/ftrl_in_git/target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar"
keep_days=16
business_line="ivr"

# 定义中断控制文件路径
INTERRUPT_FILE="train_interrupt.flag"

# 初始化环境
function init_env() {
    mkdir -p ${MODEL_OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p ./temp/
    LOGFILE="${LOG_DIR}/script_$(date +%Y%m%d).log"
    echo "log_file: ${LOG_FILE}"
    echo "oss_sample_path: ${OSS_SAMPLE_PATH}"
    echo "ftrl_jar_path: ${ftrl_jar_path}"
    touch "$INTERRUPT_FILE"
    log "INFO" "中断控制文件已创建: $INTERRUPT_FILE"

    #获取当前执行目录
    current_dir=$(pwd)
    echo "current_dir: ${current_dir}"
    # 获取lib_common.sh 脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    echo "lib_common.sh 所在的目录是: $SCRIPT_DIR"
    CURRENT_PROJ_NAME=$(basename "$PWD")
    echo "项目名称: $CURRENT_PROJ_NAME"

    cp $SCRIPT_DIR/score_kdd.py $current_dir/
    cp $SCRIPT_DIR/figure_auc_regression.py $current_dir/
    cp $SCRIPT_DIR/feishu_notifier.py $current_dir/
}

# 检查是否需要中断
function should_interrupt() {
    [[ ! -f "$INTERRUPT_FILE" ]]
}

# 日志记录函数
function log() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "[${timestamp}] [${level}] ${message}"
}

# 日期格式转换
function format_date() {
    local input_date=$1
    date -d "${input_date}" +"%Y-%m-%d"
}

function prepare_training_data(){
    local data_path=$(prepare_training_data_from_mnt $1 | tail -n 1)
    echo $data_path
}

#从本地挂载盘获取训练数据路径
function prepare_training_data_from_mnt1(){
    log "INFO" "准备训练数据..."
    local current_date="$1"
    local local_data_path=""
    local_data_path=$(find "$OSS_SAMPLE_PATH/part=$current_date" -maxdepth 1 -type f -name "*.csv" | head -n 1)
    if [[ ! -f "$local_data_path" ]]; then
        log "ERROR" "数据准备失败，样本文件不存在: $local_data_path"
        rm -rf "${tmp_dir}"
        return 1
    fi

    log "INFO" "数据准备完成: $local_data_path"
    echo "$local_data_path"
}

function prepare_training_data_from_mnt(){
    log "INFO" "准备训练数据..."
    local current_date="$1"
    local temp_dir="./temp"
    
    # 检查两个可能的路径
    local path1="$OSS_SAMPLE_PATH/part=$current_date"
    local path2="$OSS_SAMPLE_PATH/$current_date"
    local search_path=""
    
    # 判断数据存储在哪个路径下
    if [[ -d "$path1" ]]; then
        search_path="$path1"
        log "INFO" "在路径 $path1 下查找CSV文件"
    elif [[ -d "$path2" ]]; then
        search_path="$path2"
        log "INFO" "在路径 $path2 下查找CSV文件"
    else
        log "ERROR" "数据路径不存在: $path1 或 $path2"
        return 1
    fi
    
    # 查找所有匹配的CSV文件（包括子目录中的文件）
    local csv_files=()
    while IFS= read -r -d '' file; do
        csv_files+=("$file")
    done < <(find "$search_path" -type f -name "*.csv" -print0)
    
    if [[ ${#csv_files[@]} -eq 0 ]]; then
        log "ERROR" "数据准备失败，未找到CSV样本文件在路径: $search_path"
        return 1
    fi
    
    # 如果只有一个CSV文件，直接返回该文件路径
    if [[ ${#csv_files[@]} -eq 1 ]]; then
        log "INFO" "仅找到1个CSV文件，直接返回: ${csv_files[0]}"
        echo "${csv_files[0]}"
        return 0
    fi
    
    # 如果有多个CSV文件，合并它们
    local output_file="${temp_dir}/merged_data_${current_date}.csv"
    
    # 创建临时目录
    mkdir -p "$temp_dir"
    
    log "INFO" "找到 ${#csv_files[@]} 个CSV文件，开始合并..."
    
    # 使用cat命令一次性合并所有文件
    cat "${csv_files[@]}" > "$output_file"
    
    if [[ ! -f "$output_file" ]]; then
        log "ERROR" "数据合并失败，输出文件不存在: $output_file"
        rm -rf "${temp_dir}"
        return 1
    fi
    
    log "INFO" "数据准备完成，合并后的文件: $output_file"
    echo "$output_file"
}



#从 oss 下载训练数据
function prepare_training_data_from_oss() {
    log "INFO" "准备训练数据..."
    local current_date="$1"

    # 构建源路径
    if [[ "$OSS_SAMPLE_PATH" == *"ruf_sample_v1"* ]]; then
        data_path="${OSS_SAMPLE_PATH}/${current_date}"
    else
        data_path="${OSS_SAMPLE_PATH}/part=${current_date}"
    fi
    log "INFO" "数据路径: ${data_path}"

    # 创建临时目录
    local tmp_dir="./tmp_sample"
    [[ -n "${tmp_dir}" && -d "${tmp_dir}" ]] && rm -rf "${tmp_dir}" &> /dev/null
    mkdir -p "${tmp_dir}"
    if [[ $? -ne 0 ]]; then
        log "ERROR" "无法创建临时目录: ${tmp_dir}"
        return 1
    fi

    # 下载数据
    log "INFO" "从 OSS 下载数据..."
    ossutil cp "${data_path}/" "${tmp_dir}/" --recursive
    if [[ $? -ne 0 ]]; then
        log "ERROR" "OSS 数据下载失败，跳过本日训练"
        rm -rf "${tmp_dir}"
        return 1
    fi

    # 检查是否下载成功
    if ! find "${tmp_dir}" -maxdepth 1 -type f -name 'part*' | grep -q .; then
        log "ERROR" "未找到任何 part 文件，数据准备失败"
        rm -rf "${tmp_dir}"
        return 1
    fi

    # 合并或重命名文件
    mv "${tmp_dir}/part"* "${tmp_dir}/sample.dat" 2>/dev/null || {
        log "ERROR" "无法重命名 part 文件"
        rm -rf "${tmp_dir}"
        return 1
    }

    local local_data_path="${tmp_dir}/sample.dat"
    if [[ ! -f "$local_data_path" ]]; then
        log "ERROR" "数据准备失败，样本文件不存在"
        rm -rf "${tmp_dir}"
        return 1
    fi

    log "INFO" "数据准备完成: $local_data_path"
    echo "$local_data_path"
}

function execute_training() {
    local local_data_path="$1"
    local current_date="$2"

    log "INFO" "启动模型训练..."
    time java -jar -Xmx80g $ftrl_jar_path \
        -i "${local_data_path}" \
        -c "./conf" \
        -f "f" \
        -n "train-$CURRENT_PROJ_NAME-$current_date"

    if [ $? -ne 0 ]; then
        log "ERROR" "模型训练失败"
        return 1
    fi
    echo "current_dir: ${current_dir}"
}

function save_model_output() {
    local current_date="$1"
    cp "${MODEL_OUTPUT_DIR}/base" "${MODEL_OUTPUT_DIR}/base.${current_date}"
    log "INFO" "模型保存完成: ${MODEL_OUTPUT_DIR}/base.${current_date}"

    clean_old_models $current_date
}

#过滤 14 天前的老特征
function filter_old_feature() {
    local current_date="$1"
    old_date=$(date -d "${current_date} - 14 day" +"%Y-%m-%d")
    old_feature_file=${MODEL_OUTPUT_DIR}/base.${old_date}
    new_feature_file=${MODEL_OUTPUT_DIR}/base
    filted_feature_file=${MODEL_OUTPUT_DIR}/filted_feature.dat
    if [[ -f "${old_feature_file}" ]]; then
        log "INFO" "filt old feature......"
        python $SCRIPT_DIR/filter_old_feature.py "${old_feature_file}" "${new_feature_file}" > ${filted_feature_file}
        mv "${MODEL_OUTPUT_DIR}/base.tmp" "${MODEL_OUTPUT_DIR}/base"
        log "INFO" "过滤掉的老特征数: $(wc -l < ${filted_feature_file})"
    fi
}

#提取模型中非 0 权重特征
function prepare_model_file() {
    local model_file="$1"
    local filt_file="${model_file}.filt"

    if [[ ! -f "$filt_file" ]] || [[ ! -s "$filt_file" ]]; then
        log "INFO" "Filtering model file for model: $model_file"
        awk -F'\002' '{if($2!=0) print $0}' "$model_file" | sort -k2,2 -t$'\002' -r -g > "$filt_file"
    else
        log "INFO" "Filtered model file already exists and is not empty: $filt_file"
    fi
}

function post_process() { 
    awk -F'\002' '$2!=0' "${MODEL_OUTPUT_DIR}/base" | sort -t$'\002' -k2,2 -rg > "${MODEL_OUTPUT_DIR}/base.filt"
    log "INFO" "训练完成，有效记录数: $(wc -l < ${MODEL_OUTPUT_DIR}/base.filt)"
    echo "current_dir: ${current_dir}"
}

# 清理旧模型
function clean_old_models() {
    local current_date="$1"

    # 检查输入参数
    if [[ -z "$current_date" ]]; then
        log "ERROR" "缺少参数：current_date"
        exit 1
    fi

    if ! date -d "$current_date" > /dev/null 2>&1; then
        log "ERROR" "无效的日期格式: $current_date"
        return
    fi

    if [[ -z "${MODEL_OUTPUT_DIR}" ]]; then
        log "ERROR" "环境变量 MODEL_OUTPUT_DIR 未设置"
        exit 1
    fi

    if [[ ! -d "${MODEL_OUTPUT_DIR}" ]]; then
        log "ERROR" "目录不存在: ${MODEL_OUTPUT_DIR}"
        exit 1
    fi

    local cutoff_date=$(date -d "${current_date} - ${keep_days} days" +"%Y-%m-%d")
    if [[ $? -ne 0 ]]; then
        log "ERROR" "日期转换失败"
        exit 1
    fi

    log "INFO" "将清理 ${cutoff_date} 及之前的模型文件..."
    find "${MODEL_OUTPUT_DIR}" -name "base.${cutoff_date}*" -delete
    log "INFO" "已清理 ${cutoff_date} 之前的模型"
}

function train_by_date() { 
    start_date=$(format_date "$1")
    end_date=$(format_date "$2")
    current_date=${start_date}
    # 初始化环境和配置
    init_env

    # 主训练循环
    while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
        log "INFO" "================ 开始训练 ${current_date} ================"
        
        # 调用函数：准备训练数据
        local_data_path=$(prepare_training_data "$current_date" | tail -n 1)
        if [[ $? -ne 0 ]]; then
            current_date=$(date -d "${current_date} + 1 day" +"%Y-%m-%d")
            continue
        fi

        # 检查中断标志
        if should_interrupt; then
            log "WARN" "检测到中断文件不存在，准备退出..."
            break
        fi

        # 调用函数：执行训练
        execute_training "$local_data_path" "$current_date"
        if [[ $? -ne 0 ]]; then
            exit 2
        fi
        # 过滤老特征
        filter_old_feature "$current_date"
        # 调用函数：保存模型输出
        save_model_output "$current_date"

        current_date=$(date -d "${current_date} + 1 day" +"%Y-%m-%d")
    done

    # 后处理
    post_process
}

#指定训练数据训练
function train_by_sample() { 
    local local_data_path=$1
    local current_date=$2 #仅仅用于标识样本和模型
    init_env

    log "INFO" "================ 开始训练 ${local_data_path} ================"
    execute_training "$local_data_path" "$current_date"
    if [[ $? -ne 0 ]]; then
        exit 2
    fi
    save_model_output "$current_date"
}

#使用本地模型进行验证，指定模型日期和样本日期
function validation_ivr() { 
    local model_date=$1
    local sample_date=$2
    local awk_condition=$3  # 直接传递awk过滤条件语句

    init_env
    prepare_model_file "$MODEL_OUTPUT_DIR/base.${model_date}"

    model_file="${MODEL_OUTPUT_DIR}/base.${model_date}.filt"
    local_data_path=$(prepare_training_data "$sample_date" | tail -n 1)
    #--------------------------- 动态过滤样本 ------------------------------
    if [ -n "$awk_condition" ]; then
        log "INFO" "使用awk过滤条件: $awk_condition"
        # 有指定过滤条件时，使用awk进行过滤
        awk -F'\002' "$awk_condition" $local_data_path > ./temp/sample_${sample_date}.dat
        local_data_path=./temp/sample_${sample_date}.dat
    else
        log "INFO" "未指定过滤条件，使用原始数据，不对数据进行过滤"
        # 没有指定过滤条件时，直接使用原始数据，不做任何过滤
    fi
    #--------------------------- 动态过滤样本 ------------------------------
    log "INFO" "local_data_path: ${local_data_path}"

    log "INFO" "\n********************** validation begin: model=$model_file, val_date=$sample_date ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplay \
        -data "$local_data_path" \
        -conf conf/ \
        -model "${model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME-$sample_date" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$model_file, val_date=$sample_date *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
    rm -rf ./temp/sample*
}

#指定模型进行 validation
function validation_by_model() { 
    local model_date=$1
    local local_data_path=$2

    init_env
    prepare_model_file "$MODEL_OUTPUT_DIR/base.${model_date}"   
    model_file="${MODEL_OUTPUT_DIR}/base.${model_date}.filt"

    log "INFO" "\n********************** validation begin: model=$model_file, val_date=$local_data_path ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplay \
        -data "$local_data_path" \
        -conf conf/ \
        -model "${model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$model_file, val_date=$local_data_path *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
}

#使用本地模型 Validation cctm
function validation_cctm() { 
    local model_date=$1
    local sample_date=$2
    local ctr_model_dir=$3
    local cvr_model_dir=$4

    init_env

    prepare_model_file "$ctr_model_dir/base.$model_date"
    prepare_model_file "$cvr_model_dir/base.$model_date"

    ctr_model_file="${ctr_model_dir}/base.${model_date}.filt"
    cvr_model_file="${cvr_model_dir}/base.${model_date}.filt"
    echo "ctr_model_file: ${ctr_model_file}"
    echo "cvr_model_file: ${cvr_model_file}"

    local_data_path=$(prepare_training_data "$sample_date" | tail -n 1)
    log "INFO" "\n********************** validation begin: model=$ctr_model_file, $cvr_model_file, val_date=$sample_date ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplayCCTM \
        -data "$local_data_path" \
        -conf conf/ \
        -model_ctr "${ctr_model_file}" \
        -model_cvr "${cvr_model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME-$sample_date" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$ctr_model_file, $cvr_model_file, val_date=$sample_date *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
}

function validation_cctm_online() { 
    local model_date=$1
    local sample_date=$2
    init_env
    business_line="cctm"
    if [ -z "$CURRENT_PROJ_NAME" ]; then
        log "ERROR" "CURRENT_PROJ_NAME is empty"
        exit 1
    fi
    local src_ctr_model="/mnt/data/oss_dsp_algo/$business_line/$CURRENT_PROJ_NAME/ctr_v1/model/model_file_$model_date"
    echo "model_path: ${src_ctr_model}"
    if [[ ! -f "$src_ctr_model" ]]; then
        log "ERROR" "模型文件不存在，请检查"
        exit 1
    fi
    local src_cvr_model="/mnt/data/oss_dsp_algo/$business_line/$CURRENT_PROJ_NAME/cvr_v1/model/model_file_$model_date"
    echo "model_path: ${src_cvr_model}"
    if [[ ! -f "$src_cvr_model" ]]; then
        log "ERROR" "模型文件不存在，请检查"
        exit 1
    fi

    MODEL_OUTPUT_DIR="./online_model"
    local des_ctr_model=$MODEL_OUTPUT_DIR/ctr.${model_date}
    local des_cvr_model=$MODEL_OUTPUT_DIR/cvr.${model_date}
    mkdir -p $MODEL_OUTPUT_DIR
    if [[ ! -f $des_ctr_model ]]; then
        cp $src_ctr_model $des_ctr_model
    fi
    if [[ ! -f $des_cvr_model ]]; then
        cp $src_cvr_model $des_cvr_model
    fi

    prepare_model_file "$des_ctr_model"
    prepare_model_file "$des_cvr_model"

    ctr_model_file="${des_ctr_model}.filt"
    cvr_model_file="${des_cvr_model}.filt"
    local_data_path=$(prepare_training_data "$sample_date" | tail -n 1)
    log "INFO" "local_data_path: ${local_data_path}"
    log "INFO" "\n********************** validation begin: model=$ctr_model_file, $cvr_model_file, val_date=$sample_date ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplayCCTM \
        -data "$local_data_path" \
        -conf conf/ \
        -model_ctr "${ctr_model_file}" \
        -model_cvr "${cvr_model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME-$sample_date" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$ctr_model_file, $cvr_model_file, val_date=$sample_date *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
}

#使用线上的模型进行验证，样本可以自定义过滤条件
# 使用示例：
# validation_online_model "20231030" "20231029" '$7=="7762"'              # 按第7字段过滤
# validation_online_model "20231030" "20231029" '$3=="ES"'                 # 按国家代码过滤
# validation_online_model "20231030" "20231029" '$4>100'                   # 数值比较
# validation_online_model "20231030" "20231029" '$2!=""'                   # 非空判断
# validation_online_model "20231030" "20231029" '$1~/^2023/'               # 正则匹配
# validation_online_model "20231030" "20231029" '$7=="7762" && $3=="ES"'   # 复合条件
# validation_online_model "20231030" "20231029"                           # 不过滤，使用原始数据
function validation_online_model(){
    local model_date=$1
    local sample_date=$2
    local awk_condition=$3  # 直接传递awk过滤条件语句

    init_env
    business_line="ivr"
    if [ -z "$CURRENT_PROJ_NAME" ]; then
        log "ERROR" "CURRENT_PROJ_NAME is empty"
        exit 1
    fi
    local src_model_path="/mnt/data/oss_dsp_algo/$business_line/model/$CURRENT_PROJ_NAME/model_file_$model_date"
    echo "model_path: ${src_model_path}"
    if [[ ! -f "$src_model_path" ]]; then
        log "ERROR" "模型文件不存在，请检查"
        exit 1
    fi

    MODEL_OUTPUT_DIR="./online_model"
    local des_model_path="$MODEL_OUTPUT_DIR/base.${model_date}"

    mkdir -p $MODEL_OUTPUT_DIR
    if [[ ! -f $des_model_path ]]; then
        cp $src_model_path $des_model_path
    fi
    prepare_model_file "$des_model_path"
    model_file="${des_model_path}.filt"
    local_data_path=$(prepare_training_data "$sample_date" | tail -n 1)

    #--------------------------- 动态过滤样本 ------------------------------
    if [ -n "$awk_condition" ]; then
        log "INFO" "使用awk过滤条件: $awk_condition"
        # 有指定过滤条件时，使用awk进行过滤
        awk -F'\002' "$awk_condition" $local_data_path > ./temp/sample_${sample_date}.dat
        local_data_path=./temp/sample_${sample_date}.dat
    else
        log "INFO" "未指定过滤条件，使用原始数据，不对数据进行过滤"
        # 没有指定过滤条件时，直接使用原始数据，不做任何过滤
    fi
    #--------------------------- 动态过滤样本 ------------------------------
    log "INFO" "local_data_path: ${local_data_path}"

    log "INFO" "\n********************** validation begin: model=$model_file, val_date=$sample_date ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplay \
        -data "$local_data_path" \
        -conf conf/ \
        -model "${model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME-$sample_date" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$model_file, val_date=$sample_date *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
}

#使用线上模型，用本地指定的样本进行 Validation
function validation_online_by_filted_sample(){
    local model_date=$1
    local sample_date=$2
    local awk_condition=$3  # 直接传递awk过滤条件语句
    
    init_env
    business_line="ivr"
    if [ -z "$CURRENT_PROJ_NAME" ]; then
        log "ERROR" "CURRENT_PROJ_NAME is empty"
        exit 1
    fi
    local src_model_path="/mnt/data/oss_dsp_algo/$business_line/model/$CURRENT_PROJ_NAME/model_file_$model_date"
    echo "model_path: ${src_model_path}"
    if [[ ! -f "$src_model_path" ]]; then
        log "ERROR" "模型文件不存在，请检查"
        exit 1
    fi

    MODEL_OUTPUT_DIR="./online_model"
    local des_model_path="$MODEL_OUTPUT_DIR/base.${model_date}"

    mkdir -p $MODEL_OUTPUT_DIR
    if [[ ! -f $des_model_path ]]; then
        cp $src_model_path $des_model_path
    fi
    prepare_model_file "$des_model_path"
    model_file="${des_model_path}.filt"
    local_data_path=$(prepare_training_data "$sample_date" | tail -n 1)
    log "INFO" "local_data_path: ${local_data_path}"

    #--------------------------- 动态过滤样本 ------------------------------
    if [ -n "$awk_condition" ]; then
        log "INFO" "使用awk过滤条件: $awk_condition"
        # 有指定过滤条件时，使用awk进行过滤
        awk -F'\002' "$awk_condition" $local_data_path > ./temp/sample_${sample_date}.dat
        local_data_path=./temp/sample_${sample_date}.dat
    else
        log "INFO" "未指定过滤条件，使用原始数据，不对数据进行过滤"
        # 没有指定过滤条件时，直接使用原始数据，不做任何过滤
    fi
    #--------------------------- 动态过滤样本 ------------------------------

    log "INFO" "\n********************** validation begin: model=$model_file, val_date=$sample_date ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplay \
        -data "$local_data_path" \
        -conf conf/ \
        -model "${model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME-$sample_date" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$model_file, val_date=$sample_date *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
}


#上传本地模型到 oss
function upload_model() { 
    init_env
    start_date=$1
    end_date=$2
    dest_oss_path=$3
    if [ -z "$dest_oss_path" ]; then
        #判断CURRENT_PROJ_NAME是否为空
        if [ -z "$CURRENT_PROJ_NAME" ]; then
            log "ERROR" "CURRENT_PROJ_NAME is empty"
            exit 1
        fi
        dest_oss_path="oss://spark-ml-train-new/dsp_algo/$business_line/model/$CURRENT_PROJ_NAME"
    fi
    echo "目标路径: $dest_oss_path"

    # 遍历日期范围
    current_date="$start_date"
    while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
        # 构建源文件路径和目标文件名
        src_file="train_output/base.$current_date"
        dest_file="$dest_oss_path/model_file/model_file_$current_date"

        # 检查文件是否存在
        if [ -f "$src_file" ]; then
            echo "Uploading $src_file to $dest_file..."
            ossutil cp "$src_file" "$dest_file"
            if [ $? -eq 0 ]; then
                echo "Success: $src_file uploaded."
            else
                echo "Error: Failed to upload $src_file."
            fi
        else
            echo "Warning: File not found - $src_file"
        fi

        # 更新日期（使用 date 命令递增）
        current_date=$(date -I -d "$current_date + 1 day")
    done

    echo "All files processed."
    ossutil ls $dest_oss_path
}

#上传本地 conf 到 oss
function upload_conf() {
    init_env
    if [ -z "$CURRENT_PROJ_NAME" ]; then
        log "ERROR" "CURRENT_PROJ_NAME is empty"
        exit 1
    fi
    dest_oss_path="oss://spark-ml-train-new/dsp_algo/$business_line/model/$CURRENT_PROJ_NAME"
    echo "目标路径: $dest_oss_path"

    #上传 conf 目录
    ossutil cp conf $dest_oss_path/conf/ --recursive
}


# 辅助函数：将 YYYY-MM-DD-HH 格式转换为秒级时间戳
function datetime_to_timestamp() {
    local datetime_str="$1"
    # 提取年月日时
    local year=${datetime_str:0:4}
    local month=${datetime_str:5:2}
    local day=${datetime_str:8:2}
    local hour=${datetime_str:11:2}
    
    # 使用 date 命令的 --date 选项，传入标准格式
    # 注意：date -d "YYYY-MM-DD HH:MM:SS" 是标准可识别格式
    # 我们将小时设为 HH:00:00
    date -d "$year-$month-$day $hour:00:00" +"%s" 2>/dev/null
}

# 辅助函数：将秒级时间戳转换回 YYYY-MM-DD-HH 格式
function timestamp_to_datetime() {
    local timestamp="$1"
    date -d "@$timestamp" +"%Y-%m-%d-%H" 2>/dev/null
}

# 辅助函数：从 YYYY-MM-DD-HH 格式中提取日期部分 YYYY-MM-DD
function extract_date_part() {
    local datetime_str="$1"
    echo "${datetime_str%-*}" # 移除最后一个 '-' 及其后面的内容
}

# 辅助函数：从 YYYY-MM-DD-HH 格式中提取小时部分 HH
function extract_hour_part() {
    local datetime_str="$1"
    echo "${datetime_str##*-}" # 移除最后一个 '-' 及其前面的内容
}

function train_by_hour() {
    # $1: start_datetime_str (格式: YYYY-MM-DD-HH)
    # $2: end_datetime_str   (格式: YYYY-MM-DD-HH)

    local start_datetime_str="$1"
    local end_datetime_str="$2"

    # 转换起始和结束时间为时间戳以便比较
    local start_timestamp
    local end_timestamp
    start_timestamp=$(datetime_to_timestamp "$start_datetime_str")
    end_timestamp=$(datetime_to_timestamp "$end_datetime_str")

    # 检查日期时间解析是否成功
    if [[ -z "$start_timestamp" || -z "$end_timestamp" ]]; then
        log "ERROR" "无法解析起始或结束日期时间 '$start_datetime_str' 或 '$end_datetime_str'。"
        return 1
    fi
    # 初始化环境和配置
    init_env

    # 当前处理的日期时间（时间戳形式）
    local current_timestamp="$start_timestamp"

    # 主训练循环 - 按小时迭代
    while [[ "$current_timestamp" -le "$end_timestamp" ]]; do
        # 将当前时间戳转换回 YYYY-MM-DD-HH 格式
        local current_datetime_str
        current_datetime_str=$(timestamp_to_datetime "$current_timestamp")
        echo "current_datetime_str: $current_datetime_str"

        # 提取当前日期和小时部分
        local current_date
        local current_hour
        current_date=$(extract_date_part "$current_datetime_str")
        current_hour=$(extract_hour_part "$current_datetime_str")

        log "INFO" "================ 开始训练 $current_datetime_str ================"
        local_data_path=$(find "$OSS_SAMPLE_PATH/${current_date}/${current_hour}/" -maxdepth 2 -type f -name "*.csv" | head -n 1)
        echo "current_data_path: $local_data_path"
        if [[ ! -f "$local_data_path" ]]; then
            log "ERROR" "数据准备失败，样本文件不存在: $local_data_path"
            continue
        fi
        # 调用函数：执行训练 (传入数据路径和当前小时标识)
        execute_training "$local_data_path" "$current_datetime_str"
        local train_result=$?
        if [[ $train_result -ne 0 ]]; then
            log "ERROR" "执行训练失败 ($current_datetime_str)，退出。"
            exit 2
        fi

        # 调用函数：保存模型输出 (传入小时标识)
        save_model_output "$current_datetime_str"
        rm -rf temp/merged_data_*

        # 递增当前时间戳到下一个小时
        current_timestamp=$((current_timestamp + 3600)) # 3600 秒 = 1 小时
    done

    # 后处理
    post_process
}

function validation_by_hour() { 
    local model_date=$1
    local sample_date=$2

    init_env
    prepare_model_file "$MODEL_OUTPUT_DIR/base.${model_date}"
    model_file="${MODEL_OUTPUT_DIR}/base.${model_date}.filt"
    local_data_path=$(find "$OSS_SAMPLE_PATH/${sample_date:0:10}/${sample_date:11:2}/" -maxdepth 1 -type f -name "*.csv" | head -n 1)
    echo "local_data_path: $local_data_path"

    log "INFO" "\n********************** validation begin: model=$model_file, val_date=$sample_date ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplay \
        -data "$local_data_path" \
        -conf conf/ \
        -model "${model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME-$sample_date" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$model_file, val_date=$sample_date *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
    rm -rf ./temp/sample*
}

function validation_ivr_hourly() { 
    local model_date=$1
    local sample_date=$2

    init_env
    prepare_model_file "$MODEL_OUTPUT_DIR/base.${model_date}"

    model_file="${MODEL_OUTPUT_DIR}/base.${model_date}.filt"

    local_data_path=$(find "$OSS_SAMPLE_PATH/${sample_date:0:10}/${sample_date}/" -maxdepth 1 -type f -name "*.csv" | head -n 1)

    #--------------------------- 过滤样本 ------------------------------
    awk -F'\002' '{if($11!="US") print }' $local_data_path > ./temp/sample_${sample_date}.dat
    local_data_path=./temp/sample_${sample_date}.dat
    #--------------------------- 过滤样本 ------------------------------

    log "INFO" "\n********************** validation begin: model=$model_file, val_date=$sample_date ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplay \
        -data "$local_data_path" \
        -conf conf/ \
        -model "${model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME-$sample_date" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$model_file, val_date=$sample_date *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
    rm -rf ./temp/sample*
}