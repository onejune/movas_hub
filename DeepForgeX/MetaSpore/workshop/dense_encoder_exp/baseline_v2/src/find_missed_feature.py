def read_column_name_from_file(file_path):
    """
    从文件读取column_name数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        list: 特征名列表
    """
    column_name = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 假设每行格式为 "索引 特征名"，提取特征名
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    feature_name = parts[1]
                    column_name.append(feature_name)
                else:
                    column_name.append(parts[0])
    return column_name

def read_combine_schema_from_file(file_path):
    """
    从文件读取combine_schema数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        list: 组合特征列表
    """
    combine_schema = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                combine_schema.append(line)
    return combine_schema

def find_features_not_in_combine_schema(column_name, combine_schema):
    """
    找出所有在column_name中的，但是不在combine_schema中的特征
    
    Args:
        column_name: 包含所有特征的列表
        combine_schema: 包含使用到的单特征及组合特征的列表
    
    Returns:
        list: 不在combine_schema中的特征列表
    """
    # 解析combine_schema，提取所有单独的特征名
    used_features = set()
    for item in combine_schema:
        # 组合特征用#分隔，需要拆分
        features = item.split('#')
        for feature in features:
            used_features.add(feature.strip())
    
    # 找出在column_name中但不在used_features中的特征
    not_used_features = []
    for feature in column_name:
        if feature not in used_features:
            not_used_features.append(feature)
    
    return not_used_features

# 从文件读取数据
column_name_file = "./conf/column_name"  # 替换为实际的column_name文件路径
combine_schema_file = "./conf/combine_schema"  # 替换为实际的combine_schema文件路径

try:
    column_name = read_column_name_from_file(column_name_file)
    combine_schema = read_combine_schema_from_file(combine_schema_file)
    
    # 执行查找
    result = find_features_not_in_combine_schema(column_name, combine_schema)
    
    # 输出结果
    print("在column_name中但不在combine_schema中的特征:")
    for feature in result:
        print(feature)
    
    # 也可以直接输出特征名列表
    print("\n特征名列表:")
    #print(result)
    
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    print("请确保column_name.txt和combine_schema.txt文件存在")
except Exception as e:
    print(f"读取文件时发生错误: {e}")


