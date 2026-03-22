#!/bin/bash
# ============================================================================
# DEFUSE PyTorch - Criteo Benchmark
# ============================================================================
# 延迟反馈转化率预测方法对比实验
#
# 数据集: Criteo Delayed Feedback Dataset (15.9M samples)
# 方法:   Vanilla, FNW, FNC, DFM, DEFER, ES-DFM, DEFUSE, Bi-DEFUSE, Oracle
#
# 用法:
#   ./run.sh <method>           # 运行单个方法
#   ./run.sh all                # 运行所有方法（串行）
#   ./run.sh all --parallel 2   # 运行所有方法（2并行）
#   ./run.sh results            # 查看结果汇总
#
# 示例:
#   ./run.sh DEFER              # 运行 DEFER
#   ./run.sh Oracle             # 运行 Oracle baseline
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 可用方法
METHODS="Vanilla FNW FNC DFM DEFER ES-DFM DEFUSE Bi-DEFUSE Oracle"

show_help() {
    echo "用法: ./run.sh <command>"
    echo ""
    echo "Commands:"
    echo "  <method>     运行单个方法: $METHODS"
    echo "  all          运行所有方法（串行）"
    echo "  results      查看结果汇总"
    echo "  clean        清理日志和结果"
    echo ""
    echo "示例:"
    echo "  ./run.sh DEFER"
    echo "  ./run.sh all"
}

show_results() {
    echo "=== 实验结果 ==="
    python3 << 'EOF'
import json
from pathlib import Path

results = []
for f in sorted(Path("results_full").glob("*.json")):
    r = json.load(open(f))
    results.append(r)

if not results:
    print("暂无结果")
    exit(0)

results.sort(key=lambda x: x['auc'], reverse=True)
print(f"{'Rank':<5} {'Method':<12} {'AUC':<8} {'PR-AUC':<8} {'LogLoss':<8} {'Time':<8}")
print("-" * 55)
for i, r in enumerate(results, 1):
    print(f"{i:<5} {r['method']:<12} {r['auc']:.4f}   {r['pr_auc']:.4f}   {r['logloss']:.4f}   {r['time_minutes']:.1f}m")
EOF
}

run_method() {
    local method=$1
    echo "=============================================="
    echo " Running: $method"
    echo " Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
    
    mkdir -p logs_full results_full
    python run_full_parallel.py "$method" 2>&1 | tee "logs_full/${method}.log"
    
    if [ -f "results_full/${method}.json" ]; then
        echo ""
        echo "✅ $method 完成"
        python3 -c "import json; r=json.load(open('results_full/${method}.json')); print(f'   AUC={r[\"auc\"]:.4f} PR-AUC={r[\"pr_auc\"]:.4f} LogLoss={r[\"logloss\"]:.4f}')"
    else
        echo "❌ $method 失败"
    fi
}

run_all() {
    echo "运行所有方法..."
    for method in $METHODS; do
        run_method "$method"
        echo ""
    done
    show_results
}

# Main
case "${1:-help}" in
    help|--help|-h)
        show_help
        ;;
    results)
        show_results
        ;;
    all)
        run_all
        ;;
    clean)
        rm -rf logs_full/*.log results_full/*.json
        echo "已清理"
        ;;
    *)
        if echo "$METHODS" | grep -qw "$1"; then
            run_method "$1"
        else
            echo "未知方法: $1"
            echo "可用方法: $METHODS"
            exit 1
        fi
        ;;
esac
