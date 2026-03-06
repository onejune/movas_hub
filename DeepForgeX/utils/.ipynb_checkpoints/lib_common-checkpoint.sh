# File 1: lib_common.sh (公共函数库)
#!/bin/bash
set -e

# 配置区域
#OSS_SAMPLE_PATH="/mnt/data/oss_dsp_algo/ivr/sample/ruf_sample_v2/csv"
OSS_SAMPLE_PATH="oss://spark-ml-train-new/dsp_algo/ivr/sample/ruf_sample_v2/csv"
OSS_ROOT=""
MODEL_OUTPUT_DIR="./train_output"
LOG_DIR="./log"
OSSUTIL_CONFIG="./ossutilconfig"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d%H%M%S).log"
ftrl_jar_path="/mnt/data/oss_dsp_algo/tools/ftrl_jars/ftrl_maven_walter-0.0.1-SNAPSHOT.jar_v1.1"
ftrl_jar_path="/mnt/workspace/walter.wan/ftrl_in_git/target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar"

# 初始化环境
function init_env() {
    mkdir -p ${MODEL_OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    LOGFILE="${LOG_DIR}/script_$(date +%Y%m%d_%H%M%S).log"
    echo "log_file: ${LOG_FILE}"
    echo "oss_sample_path: ${OSS_SAMPLE_PATH}"
    echo "ftrl_jar_path: ${ftrl_jar_path}"
    #获取当前执行目录
    current_dir=$(pwd)
    echo "current_dir: ${current_dir}"
    # 获取lib_common.sh 脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    echo "lib_common.sh 所在的目录是: $SCRIPT_DIR"
    CURRENT_PROJ_NAME=$(basename "$PWD")
    echo "项目名称: $CURRENT_PROJ_NAME"

    cp $SCRIPT_DIR/score_kdd.py $current_dir/
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

function prepare_training_data() {
    log "INFO" "准备训练数据..."
    local current_date="$1"

    #if [[ "$OSS_SAMPLE_PATH" == *"ruf_sample_v1"* ]]; then
    #    local_data_path=$(find "$OSS_SAMPLE_PATH/$current_date" -maxdepth 1 -type f -name "*.csv")
    #else
    #    local_data_path=$(find "$OSS_SAMPLE_PATH/part=$current_date" -maxdepth 1 -type f -name "*.csv")
    #fi

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
        -n "train-$current_date"

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
    local model_date="$1"
    local model_dir="$2"
    local model_output="${model_dir}/base.${model_date}"
    local filt_file="${model_output}.filt"

    if [[ ! -f "$filt_file" ]] || [[ ! -s "$filt_file" ]]; then
        log "INFO" "Filtering model file for model: $model_output"
        awk -F'\002' '{if($2!=0) print $0}' "$model_output" | sort -k2,2 -t$'\002' -r -g > "$filt_file"
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
    local keep_days=16

    # 检查输入参数
    if [[ -z "$current_date" ]]; then
        log "ERROR" "缺少参数：current_date"
        exit 1
    fi

    if ! date -d "$current_date" > /dev/null 2>&1; then
        log "ERROR" "无效的日期格式: $current_date"
        exit 1
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

#使用本地模型进行验证，指定日期
function validation_ivr() { 
    local model_date=$1
    local sample_date=$2

    init_env
    prepare_model_file "$model_date" "$MODEL_OUTPUT_DIR"

    model_file="${MODEL_OUTPUT_DIR}/base.${model_date}.filt"
    local_data_path=$(prepare_training_data "$sample_date" | tail -n 1)

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

#指定模型进行 validation
function validation_by_model() { 
    local model_date=$1
    local local_data_path=$2

    init_env
    prepare_model_file "$model_date" "$MODEL_OUTPUT_DIR"    
    model_file="${MODEL_OUTPUT_DIR}/base.${model_date}.filt"

    log "INFO" "\n********************** validation begin: model=$model_file, val_date=$sample_path ******************************"
    time java -Xmx70g -cp $ftrl_jar_path com.mobvista.ftrl.tools.OnlineReplay \
        -data "$local_data_path" \
        -conf conf/ \
        -model "${model_file}" \
        -out "${MODEL_OUTPUT_DIR}" \
        -name "val-$CURRENT_PROJ_NAME" | tee -a "$LOGFILE"
    log "INFO" "\n********************** validation end: model=$model_file, val_date=$sample_path *****************************"
    echo "current_dir: $(pwd)"
    echo "validation completely......"
}


function validation_cctm() { 
    local model_date=$1
    local sample_date=$2
    local ctr_model=$3
    local cvr_model=$4

    prepare_model_file "$model_date" "$ctr_model"
    prepare_model_file "$model_date" "$cvr_model"

    ctr_model_file="${ctr_model}/base.${model_date}.filt"
    cvr_model_file="${cvr_model}/base.${model_date}.filt"
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

#使用线上的模型进行验证
function validation_online_model(){
    local model_dir=$1  #模型的 oss 挂载盘路径
    local model_date=$2
    local sample_date=$3

    init_env
    src_model_path="${model_dir}/model_file_${model_date}"
    echo "model_path: ${src_model_path}"
    if [[ ! -f "$src_model_path" ]]; then
        log "ERROR" "模型文件不存在，请检查"
        exit 1
    fi
    rm -rf ./conf
    cp -r ${model_dir}/conf ./
    MODEL_OUTPUT_DIR="./online_model"
    mkdir -p $MODEL_OUTPUT_DIR
    if [[ ! -f $MODEL_OUTPUT_DIR/base.${model_date} ]]; then
        cp $src_model_path $MODEL_OUTPUT_DIR/base.${model_date}
    fi

    prepare_model_file "$model_date" "$MODEL_OUTPUT_DIR"
    model_file="${MODEL_OUTPUT_DIR}/base.${model_date}.filt"
    local_data_path=$(prepare_training_data "$sample_date" | tail -n 1)
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

#上传本地模型到 oss
function upload_model() { 
    init_env
    start_date=$1
    end_date=$2
    if [ -z "$end_date" ]; then
        end_date=$start_date
    fi
    business_line="ivr"
    #判断CURRENT_PROJ_NAME是否为空
    if [ -z "$CURRENT_PROJ_NAME" ]; then
        log "ERROR" "CURRENT_PROJ_NAME is empty"
        exit 1
    fi
    dest_oss_path="oss://spark-ml-train-new/dsp_algo/$business_line/model/$CURRENT_PROJ_NAME"
    echo "目标路径: $dest_oss_path"

    # 遍历日期范围
    current_date="$start_date"
    while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
        # 构建源文件路径和目标文件名
        src_file="train_output/base.$current_date"
        dest_file="$dest_oss_path/model_file_$current_date"

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
    business_line="ivr"
    if [ -z "$CURRENT_PROJ_NAME" ]; then
        log "ERROR" "CURRENT_PROJ_NAME is empty"
        exit 1
    fi
    dest_oss_path="oss://spark-ml-train-new/dsp_algo/$business_line/model/$CURRENT_PROJ_NAME"
    echo "目标路径: $dest_oss_path"

    #上传 conf 目录
    ossutil cp conf $dest_oss_path/conf/ --recursive
}