#!/usr/bin/env python3
"""
Permutation Importance 分析工具 v1.0.0

真正的 Permutation Importance：通过打乱特征值并重新推理来评估特征重要性。

原理：
1. 计算基准 AUC（原始数据）
2. 对每个特征：打乱其值 -> 重新推理 -> 计算新 AUC
3. 重要性 = 基准 AUC - 打乱后 AUC
4. 下降越多，特征越重要

Author: Walter
Date: 2026-04-05
"""

import os
import sys
import json
import time
import argparse
import subprocess
import re
import pandas as pd
from datetime import datetime
from pathlib import Path


class PermutationImportance:
    """Permutation Importance 分析器"""
    
    VERSION = "1.0.0"
    
    def __init__(self, project_dir: str, model_date: str, sample_date: str, 
                 eval_keys: str = "business_type"):
        self.project_dir = Path(project_dir).resolve()
        self.model_date = model_date
        self.sample_date = sample_date
        self.eval_keys = eval_keys
        
        # 输出目录
        self.output_dir = self.project_dir / "output" / "permutation_importance"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志目录
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 结果文件
        self.results_file = self.output_dir / "results.json"
        self.importance_csv = self.output_dir / "importance.csv"
        
        # 加载已有结果（支持断点续跑）
        self.results = self._load_results()
        
    def _load_results(self) -> dict:
        """加载已有结果"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {"baseline": None, "features": {}}
    
    def _save_results(self):
        """保存结果"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_features(self) -> list:
        """从 combine_schema 获取特征列表"""
        schema_path = self.project_dir / "conf" / "combine_schema"
        features = []
        with open(schema_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 只取基础特征名（去掉 #demand_pkgname 等交叉后缀）
                    base_feat = line.split('#')[0]
                    if base_feat not in features:
                        features.append(base_feat)
        return features
    
    def run_validation(self, shuffle_feature: str = None) -> dict:
        """
        运行验证，返回评估结果
        
        Args:
            shuffle_feature: 要打乱的特征名，None 表示基准评估
        
        Returns:
            dict: {"auc": float, "pcoc": float, ...}
        """
        # 使用 validation_perm.sh
        script_path = self.project_dir / "validation_perm.sh"
        if not script_path.exists():
            # 创建脚本
            script_content = '''#!/bin/bash
source /mnt/workspace/walter.wan/utils/scripts/dnn_lib_common.sh
model_validation "$1" "$2" "$4" "./conf/widedeep.yaml" "$3"
'''
            script_path.write_text(script_content)
            script_path.chmod(0o755)
        
        # 构建命令
        shuffle_arg = shuffle_feature or ""
        cmd = [
            "bash", str(script_path),
            self.model_date,
            self.sample_date,
            shuffle_arg,
            self.eval_keys
        ]
        
        # 日志文件
        log_name = f"val_{shuffle_feature or 'baseline'}.log"
        log_path = self.log_dir / log_name
        
        print(f"  运行命令: {' '.join(cmd)}")
        
        try:
            # 运行并等待完成
            with open(log_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                process.wait(timeout=600)  # 10分钟超时
            
            # 解析结果
            log_content = log_path.read_text()
            return self._parse_validation_result(log_content)
            
        except subprocess.TimeoutExpired:
            process.kill()
            return {"error": "timeout"}
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_validation_result(self, output: str) -> dict:
        """解析验证输出，提取 Overall AUC"""
        # 查找 Overall 行: | val-xxx | Overall | Overall | 0.8377 | 1.0569 | ...
        pattern = r'\|\s*val-\S+\s*\|\s*Overall\s*\|\s*Overall\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
        matches = re.findall(pattern, output)
        
        if matches:
            # 取最后一个匹配
            auc, pcoc = matches[-1]
            return {
                "auc": float(auc),
                "pcoc": float(pcoc)
            }
        
        return {"error": "parse_failed"}
    
    def run_baseline(self) -> float:
        """运行基准评估"""
        if self.results.get("baseline") and "auc" in self.results["baseline"]:
            print(f"使用缓存的基准 AUC: {self.results['baseline']['auc']:.4f}")
            return self.results["baseline"]["auc"]
        
        print("运行基准评估...")
        result = self.run_validation(shuffle_feature=None)
        
        if "error" in result:
            raise RuntimeError(f"基准评估失败: {result['error']}")
        
        self.results["baseline"] = result
        self._save_results()
        
        print(f"基准 AUC: {result['auc']:.4f}")
        return result["auc"]
    
    def analyze_feature(self, feature: str, baseline_auc: float) -> dict:
        """分析单个特征的重要性"""
        # 检查缓存
        if feature in self.results.get("features", {}):
            cached = self.results["features"][feature]
            if "importance" in cached and cached["importance"] is not None:
                print(f"  {feature}: 使用缓存 (importance={cached['importance']:.4f})")
                return cached
        
        print(f"  分析特征: {feature}...")
        start_time = time.time()
        
        result = self.run_validation(shuffle_feature=feature)
        elapsed = time.time() - start_time
        
        if "error" in result:
            importance_result = {
                "feature": feature,
                "error": result["error"],
                "importance": None,
                "elapsed_seconds": elapsed
            }
        else:
            shuffled_auc = result.get("auc", 0)
            importance = baseline_auc - shuffled_auc
            
            importance_result = {
                "feature": feature,
                "baseline_auc": baseline_auc,
                "shuffled_auc": shuffled_auc,
                "importance": importance,
                "importance_pct": importance / baseline_auc * 100 if baseline_auc > 0 else 0,
                "elapsed_seconds": elapsed
            }
            print(f"    AUC: {baseline_auc:.4f} -> {shuffled_auc:.4f}, "
                  f"importance: {importance:.4f} ({importance_result['importance_pct']:.2f}%)")
        
        # 保存结果
        self.results["features"][feature] = importance_result
        self._save_results()
        
        return importance_result
    
    def run(self, features: list = None):
        """运行完整分析"""
        print("=" * 60)
        print(f"Permutation Importance 分析 v{self.VERSION}")
        print("=" * 60)
        print(f"项目目录: {self.project_dir}")
        print(f"模型日期: {self.model_date}")
        print(f"样本日期: {self.sample_date}")
        print()
        
        # 获取特征列表
        if features is None:
            features = self.get_features()
        print(f"待分析特征数: {len(features)}")
        
        # 运行基准评估
        baseline_auc = self.run_baseline()
        
        # 分析每个特征
        print(f"\n开始分析 {len(features)} 个特征...")
        
        for i, feature in enumerate(features):
            print(f"[{i+1}/{len(features)}]", end="")
            self.analyze_feature(feature, baseline_auc)
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("生成报告")
        print("=" * 60)
        
        # 转换为 DataFrame
        records = []
        for feat, data in self.results.get("features", {}).items():
            records.append({
                "feature": feat,
                "importance": data.get("importance"),
                "importance_pct": data.get("importance_pct"),
                "shuffled_auc": data.get("shuffled_auc"),
                "error": data.get("error"),
            })
        
        df = pd.DataFrame(records)
        
        # 过滤有效结果
        valid_df = df[df["importance"].notna()].copy()
        
        if len(valid_df) == 0:
            print("没有有效结果")
            return
        
        # 排序
        valid_df = valid_df.sort_values("importance", ascending=False)
        valid_df["rank"] = range(1, len(valid_df) + 1)
        
        # 分类
        p75 = valid_df["importance"].quantile(0.75)
        p50 = valid_df["importance"].quantile(0.50)
        p25 = valid_df["importance"].quantile(0.25)
        
        def classify(imp):
            if imp >= p75:
                return "HIGH"
            elif imp >= p50:
                return "MEDIUM_HIGH"
            elif imp >= p25:
                return "MEDIUM_LOW"
            elif imp > 0:
                return "LOW"
            else:
                return "NEGATIVE"
        
        valid_df["level"] = valid_df["importance"].apply(classify)
        
        # 保存 CSV
        valid_df.to_csv(self.importance_csv, index=False)
        print(f"保存结果: {self.importance_csv}")
        
        # 打印 Top 20
        print("\nTop 20 重要特征:")
        print(valid_df.head(20).to_string(index=False))
        
        # 打印 Bottom 10
        print("\nBottom 10 特征:")
        print(valid_df.tail(10).to_string(index=False))
        
        # 统计
        print("\n重要性分布:")
        print(valid_df["level"].value_counts())
        
        # 建议删除的特征
        to_remove = valid_df[valid_df["level"].isin(["LOW", "NEGATIVE"])]["feature"].tolist()
        remove_file = self.output_dir / "to_remove.txt"
        with open(remove_file, 'w') as f:
            for feat in to_remove:
                f.write(feat + '\n')
        print(f"\n建议删除的特征: {len(to_remove)} 个 -> {remove_file}")
        
        # 摘要
        summary_file = self.output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Permutation Importance 分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"版本: {self.VERSION}\n")
            f.write(f"\n")
            f.write(f"项目: {self.project_dir}\n")
            f.write(f"模型日期: {self.model_date}\n")
            f.write(f"样本日期: {self.sample_date}\n")
            f.write(f"\n")
            f.write(f"基准 AUC: {self.results.get('baseline', {}).get('auc', 'N/A')}\n")
            f.write(f"分析特征数: {len(valid_df)}\n")
            f.write(f"\n")
            f.write(f"重要性分布:\n")
            f.write(valid_df["level"].value_counts().to_string())
            f.write(f"\n\n")
            f.write(f"建议删除: {len(to_remove)} 个特征\n")
        print(f"摘要报告: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Permutation Importance 分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析所有特征
  python permutation_importance.py --project_dir ./dnn_ivr16_v2 --model_date 2026-03-02 --sample_date 2026-03-03
  
  # 只分析指定特征
  python permutation_importance.py --project_dir ./dnn_ivr16_v2 --model_date 2026-03-02 --sample_date 2026-03-03 --features country,adx,city
        """
    )
    
    parser.add_argument("--project_dir", type=str, required=True, help="项目目录")
    parser.add_argument("--model_date", type=str, required=True, help="模型日期 (YYYY-MM-DD)")
    parser.add_argument("--sample_date", type=str, required=True, help="样本日期 (YYYY-MM-DD)")
    parser.add_argument("--eval_keys", type=str, default="business_type", help="评估维度")
    parser.add_argument("--features", type=str, default=None, help="要分析的特征，逗号分隔")
    parser.add_argument("--version", action="version", version=f"%(prog)s {PermutationImportance.VERSION}")
    
    args = parser.parse_args()
    
    # 解析特征列表
    features = None
    if args.features:
        features = [f.strip() for f in args.features.split(",")]
    
    # 运行分析
    analyzer = PermutationImportance(
        project_dir=args.project_dir,
        model_date=args.model_date,
        sample_date=args.sample_date,
        eval_keys=args.eval_keys
    )
    
    analyzer.run(features=features)


if __name__ == "__main__":
    main()
