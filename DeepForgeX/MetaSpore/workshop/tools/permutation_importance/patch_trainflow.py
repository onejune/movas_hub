#!/usr/bin/env python3
"""
为项目的 base_trainFlow.py 添加 Permutation Importance 支持

用法:
    python patch_trainflow.py <project_dir>
    
这会在 _read_dataset_by_date 方法中添加特征打乱逻辑。
"""

import sys
import re
from pathlib import Path


SHUFFLE_CODE = '''
        # === Permutation Importance: 特征打乱 ===
        shuffle_feature = os.environ.get("SHUFFLE_FEATURE", None)
        if shuffle_feature and shuffle_feature in df.columns:
            MovasLogger.log(f"[Permutation Importance] 打乱特征: {shuffle_feature}")
            # 通过随机排序实现打乱
            df = df.withColumn("_pi_rand", F.rand(42))
            df = df.withColumn("_pi_row", F.row_number().over(Window.orderBy("_pi_rand")) - 1)
            
            # 提取并打乱特征值
            feature_vals = df.select("_pi_row", shuffle_feature).withColumn(
                "_pi_new_row", F.row_number().over(Window.orderBy(F.rand(43))) - 1
            )
            shuffled = feature_vals.select(
                F.col("_pi_new_row").alias("_pi_row"),
                F.col(shuffle_feature).alias(f"_pi_{shuffle_feature}")
            )
            
            # 合并回原 DataFrame
            df = df.join(shuffled, on="_pi_row", how="left")
            df = df.drop(shuffle_feature).withColumnRenamed(f"_pi_{shuffle_feature}", shuffle_feature)
            df = df.drop("_pi_rand", "_pi_row")
            MovasLogger.log(f"[Permutation Importance] 特征 {shuffle_feature} 已打乱")
        # === Permutation Importance 结束 ===
'''


def patch_file(project_dir: str):
    """为项目添加 Permutation Importance 支持"""
    project_path = Path(project_dir)
    trainflow_path = project_path / "src" / "base_trainFlow.py"
    
    if not trainflow_path.exists():
        print(f"错误: 找不到 {trainflow_path}")
        return False
    
    # 读取文件
    content = trainflow_path.read_text()
    
    # 检查是否已经打过补丁
    if "Permutation Importance" in content:
        print("文件已包含 Permutation Importance 支持，跳过")
        return True
    
    # 检查是否需要添加 Window import
    if "from pyspark.sql.window import Window" not in content:
        # 在 import 区域添加
        import_pattern = r"(from pyspark\.sql import functions as F)"
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                r"\1\nfrom pyspark.sql.window import Window",
                content
            )
        else:
            print("警告: 找不到 pyspark.sql.functions import，请手动添加 Window import")
    
    # 找到 _read_dataset_by_date 方法中 random_sample 调用之后的位置
    # 在 fillna 之前插入打乱代码
    pattern = r"(df = self\.random_sample\(df\))"
    
    if not re.search(pattern, content):
        print("错误: 找不到 random_sample 调用位置")
        return False
    
    # 插入代码
    content = re.sub(
        pattern,
        r"\1\n" + SHUFFLE_CODE,
        content
    )
    
    # 备份原文件
    backup_path = trainflow_path.with_suffix('.py.bak')
    trainflow_path.rename(backup_path)
    print(f"备份原文件: {backup_path}")
    
    # 写入新文件
    trainflow_path.write_text(content)
    print(f"已更新: {trainflow_path}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python patch_trainflow.py <project_dir>")
        sys.exit(1)
    
    project_dir = sys.argv[1]
    
    if patch_file(project_dir):
        print("\n补丁应用成功！")
        print("现在可以使用 SHUFFLE_FEATURE 环境变量来打乱特征：")
        print("  SHUFFLE_FEATURE=country ./validation.sh 2026-03-02 2026-03-03")
    else:
        print("\n补丁应用失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
