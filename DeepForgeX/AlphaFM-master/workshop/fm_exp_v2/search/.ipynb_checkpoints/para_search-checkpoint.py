import subprocess
import os
import re
from datetime import datetime, timedelta
import argparse
import time
import json

# 日期范围
start_date = datetime.strptime("2025-04-01", "%Y-%m-%d")
end_date = datetime.strptime("2025-04-16", "%Y-%m-%d")
val_date = "2025-04-17"

# 默认超参数
default_params = {
    "dim": '1,1,8',
    "init_stdev": 0.01,
    "w_l2": 100,
    "w_l1": 1,
    "v_l1": 0.1,
    "w_alpha": 0.1,
    "v_l2": 5.0,
    "v_alpha": 0.1,
    "w_beta": 1,
    "v_beta": 1.0,
}
core = 24

def evaluate_with_param(param_name, param_value, log_file, fixed_params):
    params = default_params.copy()
    params.update(fixed_params)
    params[param_name] = param_value

    para = " ".join(
        [f"-{k} {v}" for k, v in params.items()]
    ) + f" -core {core}"

    model_out = f"./output/base.{param_name}_{param_value}"

    # 日期增量训练
    for i in range((end_date - start_date).days + 1):
        cur_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        train_data = f"../../duf_outer/sample/sample_{cur_date}"
        if not os.path.exists(train_data):
            continue

        if os.path.exists(model_out):
            cmd = f"cat {train_data} | ../model_bin/fm_train {para} -m {model_out} -im {model_out}"
        else:
            cmd = f"cat {train_data} | ../model_bin/fm_train {para} -m {model_out}"
        print(f"[训练中] {cur_date}")
        st_time = time.time()
        subprocess.run(cmd, shell=True, check=True)
        ed_time = time.time()
        print(f"[训练结束] {cur_date}，耗时 {ed_time - st_time:.2f} 秒")

    # 验证
    val_data = f"../../duf_outer/sample/sample_{val_date}"
    subprocess.run(f"cat {val_data} | awk -F'\002' '{{if($13==\"COM.ZZKKO\") print $0}}' | ../model_bin/fm_predict -m {model_out} -out {model_out}.predict -core {core} -dim {params['dim']}", shell=True)
    result = subprocess.run(f"cat {model_out}.predict | python figure_auc.py", shell=True, capture_output=True, text=True)

    auc_line = [line for line in result.stdout.splitlines() if 'AUC:' in line]
    auc = float(auc_line[0].split('AUC:')[1].strip().split(',')[0])
    pcoc = float(auc_line[0].split('PCOC:')[1].strip())

    # 删除训练和验证文件
    if os.path.exists(model_out): os.remove(model_out)
    if os.path.exists(f"{model_out}.predict"): os.remove(f"{model_out}.predict")

    msg = f"{param_name} = {param_value}, {para}, AUC = {auc}, PCOC = {pcoc}"
    print(msg)
    log(msg, log_file)
    return auc, pcoc

def log(msg, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_name", type=str, required=True, help="要搜索的超参数名称，如 u_alpha")
    parser.add_argument("--values", type=str, required=True, help="枚举值列表，用逗号分隔，如 0.05,0.1,0.2")
    parser.add_argument("--fixed_params", type=str, default="{}", help="已确定的参数，JSON格式")
    args = parser.parse_args()

    param_name = args.param_name
    # values = [float(v.strip()) for v in args.values.split(",")]
    fixed_params = json.loads(args.fixed_params)
    
    
    # 特殊处理 dim 参数
    if args.param_name == "dim":
        # 使用 \002 分割多个 dim 值
        values = args.values.split("\002")  # 还原为 ["1,1,1", "1,1,4", ...]
    else:
        values = [float(v) for v in args.values.split(",")]
    
    fixed_params = json.loads(args.fixed_params)
    
    # 验证 dim 格式
    if args.param_name == "dim":
        for val in values:
            parts = val.split(",")
            if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
                raise ValueError(f"无效的 dim 值: {val} (应为 a,b,c 格式)")

    log_file = f"search_{param_name}_log.txt"
    log(f"values:{values}", log_file)
    best_auc = -1.0
    best_val = None

    for val in values:
        try:
            auc, pcoc = evaluate_with_param(param_name, val, log_file, fixed_params)
            if auc > best_auc:
                best_auc = auc
                best_val = val
                log(f"更新最佳：{param_name} = {val}, AUC = {auc}, PCOC = {pcoc}", log_file)
        except Exception as e:
            msg = f"[出错] {param_name} = {val} 时发生错误：{e}"
            print(msg)
            log(msg, log_file)

    final_msg = f"最优 {param_name}: {best_val}, 最佳 AUC: {best_auc}"
    print(final_msg)
    log(final_msg, log_file)
