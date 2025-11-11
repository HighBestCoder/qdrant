#!/bin/bash
# VDE vs Native HNSW Comparison Benchmark
# 对比 VDE 和原生 Qdrant HNSW 的性能

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== VDE vs Native HNSW Performance Comparison ===${NC}"
echo ""

# 配置
DATASET="${1:-glove-100-angular}"
PARALLEL="${2:-8}"

echo "Dataset: $DATASET"
echo "Parallel requests: $PARALLEL"
echo ""

# 测试配置对
declare -a TESTS=(
    "qdrant-vde-m16-ef100:qdrant-m-16-ef-100"
    "qdrant-vde-m32-ef256:qdrant-m-32-ef-256"
)

cd /src/db/vector-db-benchmark

# 确保依赖已安装
if [ ! -d ".venv" ]; then
    echo "Installing dependencies..."
    poetry install
fi

# 运行对比测试
for test_pair in "${TESTS[@]}"; do
    IFS=':' read -r vde_config native_config <<< "$test_pair"
    
    echo -e "${BLUE}=== Testing: $vde_config vs $native_config ===${NC}"
    echo ""
    
    # 测试 VDE
    echo -e "${YELLOW}[1/2] Testing VDE configuration: $vde_config${NC}"
    poetry run python run.py \
        --engines "$vde_config" \
        --datasets "$DATASET" \
        --host localhost
    
    echo ""
    
    # 测试原生
    echo -e "${YELLOW}[2/2] Testing Native configuration: $native_config${NC}"
    poetry run python run.py \
        --engines "$native_config" \
        --datasets "$DATASET" \
        --host localhost
    
    echo ""
    echo -e "${GREEN}✓ Comparison complete for $test_pair${NC}"
    echo "=================================================="
    echo ""
done

# 生成对比报告
echo -e "${GREEN}=== Generating Comparison Report ===${NC}"
echo ""

RESULTS_DIR="/src/db/vector-db-benchmark/results"
LATEST_RESULTS=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -4)

echo "Recent test results:"
echo "$LATEST_RESULTS"
echo ""

# 创建简单的对比表格
echo "Performance Summary:"
echo "==================="
echo ""

for result_file in $LATEST_RESULTS; do
    config_name=$(basename "$result_file" .json | cut -d'-' -f1-3)
    
    # 提取关键指标（需要 jq）
    if command -v jq &> /dev/null; then
        mean_time=$(jq -r '.results[0].mean_time // "N/A"' "$result_file" 2>/dev/null || echo "N/A")
        rps=$(jq -r '.results[0].rps // "N/A"' "$result_file" 2>/dev/null || echo "N/A")
        recall=$(jq -r '.results[0].mean_precisions // "N/A"' "$result_file" 2>/dev/null || echo "N/A")
        
        printf "%-30s | Time: %-10s | RPS: %-10s | Recall: %-10s\n" \
            "$config_name" "$mean_time" "$rps" "$recall"
    else
        echo "Config: $config_name (install jq for detailed metrics)"
    fi
done

echo ""
echo -e "${GREEN}=== Comparison Complete ===${NC}"
echo ""
echo "Full results available in: $RESULTS_DIR"
echo "View online dashboard: https://qdrant.tech/benchmarks/ (after uploading)"
