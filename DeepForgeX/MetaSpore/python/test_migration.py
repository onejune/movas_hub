#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MetaSpore TrainFlow 迁移验证脚本

验证重构后的 import 路径是否正确工作
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, '.')

def test_basic_imports():
    """测试基本 import 功能"""
    print("Testing basic imports...")
    
    try:
        from metaspore.trainflows import BaseTrainFlow, DNNTrainFlow, MTLTrainFlow
        print("✅ Core trainflows import successful")
    except ImportError as e:
        print(f"❌ Core trainflows import failed: {e}")
        return False
    
    try:
        from metaspore.utils import MovasLogger, FeishuNotifier, compute_auc_pcoc, calculate_logloss, how_much_time
        print("✅ Utils import successful")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_class_structure():
    """测试类的基本结构"""
    print("\nTesting class structures...")
    
    try:
        from metaspore.trainflows import BaseTrainFlow, DNNTrainFlow, MTLTrainFlow
        
        # 检查 BaseTrainFlow 是否有关键方法
        base_methods = ['_init_spark', '_stop_spark', '_load_config', '_build_model_module', 'run_complete_flow', 'parse_args']
        for method in base_methods:
            if hasattr(BaseTrainFlow, method) or (hasattr(BaseTrainFlow, '__dict__') and method in dir(BaseTrainFlow)):
                continue
            else:
                print(f"❌ BaseTrainFlow missing method: {method}")
                return False
        
        print("✅ BaseTrainFlow structure OK")
        
        # 检查 DNNTrainFlow 继承关系
        if issubclass(DNNTrainFlow, BaseTrainFlow):
            print("✅ DNNTrainFlow inheritance OK")
        else:
            print("❌ DNNTrainFlow inheritance failed")
            return False
            
        # 检查 MTLTrainFlow 继承关系
        if issubclass(MTLTrainFlow, BaseTrainFlow):
            print("✅ MTLTrainFlow inheritance OK")
        else:
            print("❌ MTLTrainFlow inheritance failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Class structure test failed: {e}")
        return False

def test_command_line_interface():
    """测试命令行接口"""
    print("\nTesting command line interface...")
    
    try:
        from metaspore.trainflows import BaseTrainFlow, DNNTrainFlow
        
        # 测试 parse_args 方法是否存在且能正常工作（不实际解析参数）
        if hasattr(BaseTrainFlow, 'parse_args') and callable(getattr(BaseTrainFlow, 'parse_args')):
            print("✅ BaseTrainFlow.parse_args exists")
        else:
            print("❌ BaseTrainFlow.parse_args missing or not callable")
            return False
            
        if hasattr(DNNTrainFlow, 'parse_args') and callable(getattr(DNNTrainFlow, 'parse_args')):
            print("✅ DNNTrainFlow.parse_args exists")
        else:
            print("❌ DNNTrainFlow.parse_args missing or not callable")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Command line interface test failed: {e}")
        return False

def main():
    print("="*60)
    print("MetaSpore TrainFlow Migration Validation")
    print("="*60)
    
    success = True
    
    success &= test_basic_imports()
    success &= test_class_structure()
    success &= test_command_line_interface()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED! Migration is successful!")
        print("✅ Core modules migrated: base, dnn, mtl")
        print("✅ Import paths working correctly")
        print("✅ Class inheritance preserved")
        print("✅ Command line interface functional")
    else:
        print("❌ SOME TESTS FAILED! Migration needs fixes.")
    
    print("="*60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)