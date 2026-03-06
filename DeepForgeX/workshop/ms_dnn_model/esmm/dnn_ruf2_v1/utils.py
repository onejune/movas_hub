import os
from datetime import datetime, timedelta

def delete_old_model(directory, specified_date_str, days = 14):
    # 将指定日期字符串转换为日期对象
    try:
        specified_date = datetime.strptime(specified_date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Invalid date format: {specified_date_str}. Please use YYYY-MM-DD format.")
        return
    
    # 计算 14 天前的日期
    target_date = specified_date - timedelta(days=days)
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    # 构建目标文件名
    target_filename = f"model_{target_date_str}"
    file_path = os.path.join(directory, target_filename)
    
    # 检查文件是否存在并删除
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {file_path} (Date: {target_date_str})")
    else:
        print(f"File not found: {file_path}")




