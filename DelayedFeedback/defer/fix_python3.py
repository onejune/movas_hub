#!/usr/bin/env python3
"""
自动修复 Python 2 -> Python 3 兼容性问题
"""
import os
import re

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')

def fix_print_statements(content):
    """修复 print 语句为 print() 函数"""
    # 匹配 print 'xxx' 或 print xxx 形式
    # 但不匹配 print(xxx) 形式
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # 跳过已经是 print() 形式的
        if re.match(r'^\s*print\s*\(', line):
            fixed_lines.append(line)
            continue
        
        # 匹配 print 'xxx' 形式
        match = re.match(r'^(\s*)print\s+(.+)$', line)
        if match:
            indent = match.group(1)
            content_part = match.group(2).strip()
            # 处理多个参数的情况
            fixed_lines.append(f'{indent}print({content_part})')
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_gpu_config(content):
    """修复 GPU 配置，支持无 GPU 环境"""
    old_gpu_config = '''physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], enable=True
)
if len(physical_devices) > 1:
    tf.config.experimental.set_memory_growth(
        physical_devices[1], enable=True
    )'''
    
    new_gpu_config = '''physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")
if len(physical_devices) > 1:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)
    except RuntimeError as e:
        print(f"GPU 1 memory growth setting failed: {e}")'''
    
    content = content.replace(old_gpu_config, new_gpu_config)
    
    # 另一种形式
    old_gpu_config2 = '''physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

physical_devices = tf.config.experimental.list_physical_devices('GPU') #tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(
#    physical_devices[0], enable=True
#)
if len(physical_devices) > 1:
    tf.config.experimental.set_memory_growth(
        physical_devices[1], enable=True
    )'''
    
    new_gpu_config2 = '''physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")
if len(physical_devices) > 1:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)
    except RuntimeError as e:
        print(f"GPU 1 memory growth setting failed: {e}")'''
    
    content = content.replace(old_gpu_config2, new_gpu_config2)
    
    return content

def fix_file(filepath):
    """修复单个文件"""
    print(f"修复文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复 print 语句
    content = fix_print_statements(content)
    
    # 修复 GPU 配置
    content = fix_gpu_config(content)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  已修改: {filepath}")
        return True
    else:
        print(f"  无需修改: {filepath}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("开始修复 Python 2 -> Python 3 兼容性问题")
    print("=" * 60)
    
    py_files = [
        'data.py',
        'loss.py',
        'main.py',
        'metrics.py',
        'models.py',
        'pretrain.py',
        'stream_train_test.py',
        'test.py',
        'utils.py'
    ]
    
    modified_count = 0
    for filename in py_files:
        filepath = os.path.join(SRC_DIR, filename)
        if os.path.exists(filepath):
            if fix_file(filepath):
                modified_count += 1
        else:
            print(f"警告: 文件不存在 {filepath}")
    
    print("=" * 60)
    print(f"修复完成，共修改 {modified_count} 个文件")
    print("=" * 60)

if __name__ == '__main__':
    main()
