#!/usr/bin/env python3
"""
Permutation Importance 特征重要性分析

用法:
    # 分析指定特征
    python feature_importance.py \
        --conf ./workshop/ctr/dnn_ivr16_v2/conf/widedeep.yaml \
        --model_date 2026-03-02 \
        --sample_date 2026-03-03 \
        --features country,adx,city

    # 分析全部特征（需指定 schema 文件）
    python feature_importance.py \
        --conf ./conf/widedeep.yaml \
        --model_date 2026-03-02 \
        --sample_date 2026-03-03 \
        --schema ./conf/combine_schema

    # 分组评估（同时打乱多个特征）
    python feature_importance.py \
        --conf ./conf/widedeep.yaml \
        --model_date 2026-03-02 \
        --sample_date 2026-03-03 \
        --group "realtime:huf_*,ruf_*" \
        --group "inner:duf_inner_*" \
        --group "outer:duf_outer_*"

Author: Walter
Date: 2026-04-06
"""

import os
import re
import sys
import json
import fnmatch
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
METASPORE_ROOT = SCRIPT_DIR.parent                    # .../MetaSpore
DEEPFORGEX_ROOT = METASPORE_ROOT.parent               # .../DeepForgeX
METASPORE_PYTHON = METASPORE_ROOT / "python"
TRAINFLOW_SCRIPT = DEEPFORGEX_ROOT / "utils" / "trainflows" / "dnn_trainFlow.py"
PYTHON_ENV = "/root/anaconda3/envs/spore/bin/python"


def init_env():
    """初始化环境，复制依赖文件"""
    import shutil
    
    utils_dir = DEEPFORGEX_ROOT / "utils"
    
    # 复制 movas_logger.py 到 metaspore 目录
    src = utils_dir / "tools" / "movas_logger.py"
    dst = METASPORE_PYTHON / "metaspore" / "movas_logger.py"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)


def get_env() -> Dict[str, str]:
    """运行环境变量"""
    env = os.environ.copy()
    
    # PYTHONPATH: MetaSpore/python + utils/tools (含 movas_logger)
    utils_tools = DEEPFORGEX_ROOT / "utils" / "tools"
    pythonpath = f"{METASPORE_PYTHON}:{utils_tools}"
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    
    env["PYSPARK_PYTHON"] = PYTHON_ENV
    env["PYSPARK_DRIVER_PYTHON"] = PYTHON_ENV
    return env


def load_features_from_schema(schema_path: str) -> List[str]:
    """从 combine_schema 读取特征列表"""
    features = []
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                base_feat = line.split('#')[0]
                if base_feat not in features:
                    features.append(base_feat)
    return features


def parse_groups(group_args: List[str], all_features: List[str]) -> Dict[str, List[str]]:
    """
    解析分组参数
    
    格式: "group_name:pattern1,pattern2,..."
    支持通配符: * 匹配任意字符
    
    返回: {group_name: [matched_features]}
    """
    groups = {}
    for g in group_args:
        if ':' not in g:
            print(f"警告: 无效的分组格式 '{g}'，应为 'name:pattern1,pattern2'")
            continue
        
        name, patterns_str = g.split(':', 1)
        patterns = [p.strip() for p in patterns_str.split(',')]
        
        matched = []
        for feat in all_features:
            for pattern in patterns:
                if fnmatch.fnmatch(feat, pattern):
                    if feat not in matched:
                        matched.append(feat)
                    break
        
        if matched:
            groups[name] = matched
            print(f"分组 '{name}': 匹配 {len(matched)} 个特征")
        else:
            print(f"警告: 分组 '{name}' 没有匹配到任何特征")
    
    return groups


def run_validation(
    conf: str,
    model_date: str,
    sample_date: str,
    eval_keys: str,
    name: str = "pi_test",
    shuffle_feature: Optional[str] = None,
    workdir: Optional[str] = None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """运行一次验证"""
    cmd = [
        PYTHON_ENV,
        str(TRAINFLOW_SCRIPT),
        "--conf", conf,
        "--name", name,
        "--validation", "True",
        "--model_date", model_date,
        "--sample_date", sample_date,
        "--eval_keys", eval_keys,
    ]
    if shuffle_feature:
        cmd.extend(["--shuffle_feature", shuffle_feature])
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                proc = subprocess.run(
                    cmd, cwd=workdir, env=get_env(),
                    stdout=f, stderr=subprocess.STDOUT,
                    timeout=1800
                )
        else:
            proc = subprocess.run(
                cmd, cwd=workdir, env=get_env(),
                capture_output=True, text=True,
                timeout=1800
            )
            log_content = proc.stdout
        
        # 解析结果
        content = open(log_file).read() if log_file else log_content
        pattern = r'\|\s*val-\S+\s*\|\s*Overall\s*\|\s*Overall\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
        matches = re.findall(pattern, content)
        
        if matches:
            auc, pcoc = matches[-1]
            return {"auc": float(auc), "pcoc": float(pcoc), "error": None}
        return {"auc": None, "pcoc": None, "error": "parse_failed"}
        
    except subprocess.TimeoutExpired:
        return {"auc": None, "pcoc": None, "error": "timeout"}
    except Exception as e:
        return {"auc": None, "pcoc": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Permutation Importance 特征重要性分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析指定特征
  python feature_importance.py \\
      --conf ./conf/widedeep.yaml \\
      --model_date 2026-03-02 \\
      --sample_date 2026-03-03 \\
      --features country,adx,city

  # 分析全部特征
  python feature_importance.py \\
      --conf ./conf/widedeep.yaml \\
      --model_date 2026-03-02 \\
      --sample_date 2026-03-03 \\
      --schema ./conf/combine_schema

  # 分组评估（同时打乱组内所有特征）
  python feature_importance.py \\
      --conf ./conf/widedeep.yaml \\
      --model_date 2026-03-02 \\
      --sample_date 2026-03-03 \\
      --schema ./conf/combine_schema \\
      --group "realtime:huf_*,ruf_*" \\
      --group "inner:duf_inner_*" \\
      --group "outer:duf_outer_*"
        """
    )
    
    parser.add_argument("--conf", "-c", required=True, help="配置文件 yaml")
    parser.add_argument("--model_date", "-m", required=True, help="模型日期 YYYY-MM-DD")
    parser.add_argument("--sample_date", "-s", required=True, help="样本日期 YYYY-MM-DD")
    parser.add_argument("--features", "-f", help="特征列表，逗号分隔")
    parser.add_argument("--schema", help="combine_schema 文件")
    parser.add_argument("--group", "-g", action="append", default=[], 
                        help="分组评估，格式: 'name:pattern1,pattern2' 支持通配符*")
    parser.add_argument("--eval_keys", "-e", default="business_type", help="评估维度")
    parser.add_argument("--output", "-o", default="./pi_output", help="输出目录")
    parser.add_argument("--workdir", "-w", help="工作目录，默认 conf 文件所在目录的上级")
    parser.add_argument("--clear", action="store_true", help="清除缓存重新开始")
    
    args = parser.parse_args()
    
    # 初始化环境
    init_env()
    
    # 解析参数
    conf = Path(args.conf).resolve()
    if not conf.exists():
        print(f"错误: 配置文件不存在: {conf}")
        sys.exit(1)
    
    workdir = args.workdir or str(conf.parent.parent)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 获取特征列表
    all_features = []
    if args.schema:
        all_features = load_features_from_schema(args.schema)
    
    # 确定评估目标：分组模式 or 单特征模式
    if args.group:
        # 分组模式
        if not all_features:
            print("错误: 分组模式需要指定 --schema")
            sys.exit(1)
        groups = parse_groups(args.group, all_features)
        if not groups:
            print("错误: 没有有效的分组")
            sys.exit(1)
        mode = "group"
        targets = groups  # {group_name: [features]}
    elif args.features:
        # 单特征模式
        features = [f.strip() for f in args.features.split(",")]
        mode = "single"
        targets = {f: [f] for f in features}  # 每个特征单独一组
    elif args.schema:
        # 全特征模式
        mode = "single"
        targets = {f: [f] for f in all_features}
    else:
        print("错误: 必须指定 --features, --schema, 或 --group")
        sys.exit(1)
    
    # checkpoint
    ckpt_file = output_dir / "checkpoint.json"
    if args.clear and ckpt_file.exists():
        ckpt_file.unlink()
    
    checkpoint = {}
    if ckpt_file.exists():
        checkpoint = json.loads(ckpt_file.read_text())
    
    def save_ckpt():
        ckpt_file.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False))
    
    # ============================================================
    # 开始分析
    # ============================================================
    print("=" * 60)
    print("Permutation Importance 特征重要性分析")
    print("=" * 60)
    print(f"配置文件: {conf}")
    print(f"模型日期: {args.model_date}")
    print(f"样本日期: {args.sample_date}")
    print(f"评估模式: {'分组' if mode == 'group' else '单特征'}")
    print(f"评估目标: {len(targets)} 个")
    print(f"输出目录: {output_dir}")
    print()
    
    # 从 conf 路径提取项目名
    proj_name = conf.parent.parent.name
    
    # 1. 基准评估
    print("[Baseline] 运行基准评估...", end=" ", flush=True)
    if "baseline" in checkpoint and checkpoint["baseline"].get("auc"):
        baseline = checkpoint["baseline"]
        print(f"(缓存) AUC: {baseline['auc']:.4f}")
    else:
        baseline = run_validation(
            str(conf), args.model_date, args.sample_date, args.eval_keys,
            name=proj_name, workdir=workdir, log_file=str(log_dir / "baseline.log")
        )
        if baseline["error"]:
            print(f"失败: {baseline['error']}")
            sys.exit(1)
        checkpoint["baseline"] = baseline
        save_ckpt()
        print(f"AUC: {baseline['auc']:.4f}, PCOC: {baseline['pcoc']:.4f}")
    
    baseline_auc = baseline["auc"]
    
    # 2. 分析每个目标（特征或分组）
    print(f"\n分析 {len(targets)} 个{'分组' if mode == 'group' else '特征'}:\n")
    
    results = []
    for i, (target_name, target_features) in enumerate(targets.items()):
        # shuffle_feature 参数：多个特征用逗号分隔
        shuffle_str = ",".join(target_features)
        display_name = target_name if mode == "group" else target_name
        
        print(f"[{i+1}/{len(targets)}] {display_name}", end="")
        if mode == "group":
            print(f" ({len(target_features)} 特征)", end="")
        print("...", end=" ", flush=True)
        
        cache_key = f"target_{target_name}"
        if cache_key in checkpoint and checkpoint[cache_key].get("shuffled_auc"):
            r = checkpoint[cache_key]
            print(f"(缓存) AUC: {r['shuffled_auc']:.4f}, Δ: {r['importance']:.4f}")
            results.append(r)
            continue
        
        # 安全的日志文件名
        safe_name = re.sub(r'[^\w\-]', '_', target_name)
        res = run_validation(
            str(conf), args.model_date, args.sample_date, args.eval_keys,
            name=proj_name, shuffle_feature=shuffle_str, workdir=workdir,
            log_file=str(log_dir / f"{safe_name}.log")
        )
        
        if res["error"]:
            print(f"错误: {res['error']}")
            r = {
                "name": target_name,
                "features": target_features,
                "feature_count": len(target_features),
                "shuffled_auc": None,
                "importance": None,
                "error": res["error"]
            }
        else:
            imp = baseline_auc - res["auc"]
            r = {
                "name": target_name,
                "features": target_features,
                "feature_count": len(target_features),
                "shuffled_auc": res["auc"],
                "shuffled_pcoc": res["pcoc"],
                "importance": imp,
                "importance_pct": imp / baseline_auc * 100,
                "error": None
            }
            print(f"AUC: {res['auc']:.4f}, Δ: {imp:.4f} ({r['importance_pct']:.2f}%)")
        
        checkpoint[cache_key] = r
        save_ckpt()
        results.append(r)
    
    # 3. 生成报告
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    
    # 排序
    valid = [r for r in results if r.get("importance") is not None]
    valid.sort(key=lambda x: x["importance"], reverse=True)
    
    # 保存 CSV
    csv_file = output_dir / "importance.csv"
    with open(csv_file, 'w') as f:
        f.write("rank,name,feature_count,shuffled_auc,importance,importance_pct\n")
        for i, r in enumerate(valid, 1):
            f.write(f"{i},{r['name']},{r['feature_count']},{r['shuffled_auc']:.4f},{r['importance']:.4f},{r['importance_pct']:.2f}\n")
    
    # 打印 Top 10
    print(f"\nBaseline AUC: {baseline_auc:.4f}\n")
    print(f"Top 10 重要{'分组' if mode == 'group' else '特征'}:")
    print("-" * 60)
    for i, r in enumerate(valid[:10], 1):
        name_display = r['name'][:35]
        if mode == "group":
            print(f"{i:2}. {name_display:35} ({r['feature_count']:3} 特征) Δ={r['importance']:.4f} ({r['importance_pct']:.2f}%)")
        else:
            print(f"{i:2}. {name_display:35} Δ={r['importance']:.4f} ({r['importance_pct']:.2f}%)")
    
    print(f"\n结果保存: {csv_file}")
    
    # 分组模式额外输出
    if mode == "group":
        detail_file = output_dir / "group_details.json"
        with open(detail_file, 'w') as f:
            json.dump(valid, f, indent=2, ensure_ascii=False)
        print(f"分组详情: {detail_file}")


if __name__ == "__main__":
    main()
