ip=52.2.100.149
function log {
    local text;local logtype
    logfile=log/log.`date +%Y%m%d`
    logtype=$1
    text=$2
    message="[`date +'%F %H:%M:%S'`] [$1] $2"
    case $logtype in 
        ERROR)
            echo -e "\033[31m${message}\033[0m" | tee -a $logfile;;
        INFO)
            echo -e "\033[32m${message}\033[0m" | tee -a $logfile;;
        WARN)
            echo -e "\033[33m${message}\033[0m" | tee -a $logfile;;
    esac
}
function checkFileMd5() {
    local md5file=$1
    local file=$2
    if [ ! -f "$md5file" ] || [ ! -f "$file" ] ; then
        return 1
    fi
    local md5_1=`head -n 1 "$md5file" | awk '{print $1}'`
    local md5_2=`md5sum "$file" | awk '{print $1}'`
    if [ "$md5_1" == "$md5_2" ]; then
        return 0
    else
        return 1
        log WARN "$file $md5_1 != $md5_2" "$LOG_FILE"
    fi
}
function email(){
    level="$1"
    subject="$2"
    content="$3"
    python /home/mobdev/shenlei/online/emailTools.py "$level" "$subject" "$content ip:$ip"

}
function checkReady(){
    train_data_path=$1
    success_flag=$train_data_path/_SUCCESS
    i=0
    for i in  `seq 1 30`
    do
        aws s3 ls $success_flag
        if [ $? -ne 0 ]; then
            log "WARN" "$success_flag is not ready!"
            sleep 1m
            continue
        fi
        break
    done
    aws s3 ls $success_flag
    if [ $? -ne 0 ]; then
        log "ERROR" "$success_flag is not ready!"
        exit 1
    else
        log "INFO" "$success_flag is ready!"
    fi
}
function check_lines(){
    f=$1
    min_num=$2
    num=`wc -l $f | awk '{print $1}'`
    if [ $num -lt $min_num ]; then
        return 1
    fi
    return 0
}

