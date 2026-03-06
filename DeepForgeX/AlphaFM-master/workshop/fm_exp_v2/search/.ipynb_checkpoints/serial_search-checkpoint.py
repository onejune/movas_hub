import subprocess
import json
import time

# 超参数搜索空间
param_grid = {
    # 核心学习率参数（FTRL的核心超参）
    "w_alpha": [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    "v_alpha": [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    
    # 模型结构参数
    # "-dim": ["1,1,1", "1,1,4", "1,1,8", "1,1,16", "1,1,32", "1,0,8", "0,1,8"],
    
    # 主正则化参数（L2优先于L1）
    "w_l2": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    "v_l2": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    
    # 初始化参数（二阶特征交互的关键）
    "init_stdev": [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    
    # 学习率衰减参数
    "w_beta": [0.0, 0.3, 0.5, 1.0, 2.0, 5.0],
    "v_beta": [0.0, 0.3, 0.5, 1.0, 2.0, 5.0],
    
    # 稀疏化参数（L1建议后期单独调）
    "w_l1": [0.0, 0.001, 0.003, 0.01, 0.03, 0.1],
    "v_l1": [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
}
# 初始默认参数
current_best_params = {
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

# 搜索函数：顺序搜索一个参数并返回最佳值
def run_search(param_name, values, fixed_params):
    # val_str = values if param_name == "dim" else ",".join(map(str, values))
    # val_str = ",".join(str(v) for v in values)

    if param_name == "dim":
        val_str = "\002".join(values)
        # 将整个列表转换为JSON字符串格式（保留每个dim值的完整性）
    else:
        val_str = ",".join(map(str, values))  # 普通参数格式："0.01,0.1,0.2"
    
    fixed_json = json.dumps(fixed_params)
    print(f"[调试] values: {values}")
    print(f"[调试] fixed_json: {fixed_json}")  # 检查 dim 是否仍是 "1,1,1"
    print(f"[调试] val_str: {val_str}")
    cmd = f"python para_search.py --param_name {param_name} --values {val_str} --fixed_params '{fixed_json}'"
    print(f"[开始搜索] {param_name}（固定参数：{fixed_params}）\n{'='*60}")
    # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # result = subprocess.run(cmd, shell=True)
    result = subprocess.run(
        f"{cmd} | tee /dev/tty",  # 同时输出到终端和stdout
        shell=True,
        capture_output=True,      # 捕获输出
        text=True,
    )
    best_val = None
    for line in result.stdout.splitlines():
        print(line)
        if line.startswith("最优"):
            try:
                # 特殊处理 dim 参数（保留字符串）
                if param_name == "dim":
                    best_val = line.split(":")[1].split(" ")[0].strip()
                else:
                    best_val = float(line.split(":")[1].split(",")[0].strip())
            except:
                pass

    if best_val is None:
        print("[警告] 没有解析到最优值，可能para_search输出格式不符合预期。\n输出如下：")
        print(result.stdout)
    return best_val


if __name__ == "__main__":
    for param_name, values in param_grid.items():
        best_val = run_search(param_name, values, current_best_params)
        if best_val is not None:
            current_best_params[param_name] = best_val
            print(f"[更新参数] {param_name} = {best_val}")

    print("\n[全部搜索完成] 当前最优参数组合：")
    for k, v in current_best_params.items():
        print(f"{k} = {v}")
