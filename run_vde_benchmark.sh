#!/bin/bash
# VDE Performance Benchmark Script
# 用于测试 Qdrant + VDE 的性能

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Qdrant VDE Performance Benchmark ===${NC}"
echo ""

# 检查环境
check_environment() {
    echo -e "${YELLOW}[1/5] Checking environment...${NC}"
    
    # 检查 Qdrant 是否编译
    if [ ! -f "/src/db/qdrant/target/release/qdrant" ]; then
        echo -e "${RED}Error: Qdrant not built in release mode${NC}"
        echo "Building Qdrant..."
        cd /src/db/qdrant
        cargo build --release
    fi
    
    # 检查 VDE 库
    if [ ! -f "/src/db/vde/build/lib/libvde.a" ]; then
        echo -e "${RED}Error: VDE library not found${NC}"
        exit 1
    fi
    
    # 检查 Btrieve
    if [ ! -f "/usr/local/actianzen/lib64/libbtrieveCpp.so" ]; then
        echo -e "${RED}Error: Btrieve library not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Environment check passed${NC}"
    echo ""
}

# 启动 Qdrant 服务器
start_qdrant() {
    echo -e "${YELLOW}[2/5] Starting Qdrant server with VDE...${NC}"
    
    # 停止现有实例
    pkill -f "qdrant" || true
    sleep 2
    
    # 设置环境变量
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    export LD_LIBRARY_PATH="/src/db/vde/build/lib:/usr/local/actianzen/lib64:$LD_LIBRARY_PATH"
    
    # 启动 Qdrant
    cd /src/db/qdrant
    nohup ./target/release/qdrant > /tmp/qdrant-vde.log 2>&1 &
    QDRANT_PID=$!
    
    # 等待启动
    echo "Waiting for Qdrant to start..."
    for i in {1..30}; do
        if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Qdrant started successfully (PID: $QDRANT_PID)${NC}"
            echo ""
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}Error: Qdrant failed to start${NC}"
    cat /tmp/qdrant-vde.log
    exit 1
}

# 运行基准测试
run_benchmark() {
    echo -e "${YELLOW}[3/5] Running benchmark...${NC}"
    
    cd /src/db/vector-db-benchmark
    
    # 安装依赖（如果需要）
    if [ ! -d ".venv" ]; then
        echo "Installing dependencies..."
        poetry install
    fi
    
    # 设置数据集（默认使用小数据集进行快速测试）
    DATASET="${1:-glove-100-angular}"
    CONFIG="${2:-qdrant-vde-default}"
    
    echo "Dataset: $DATASET"
    echo "Config: $CONFIG"
    echo ""
    
    # 运行测试
    poetry run python run.py \
        --engines "$CONFIG" \
        --datasets "$DATASET" \
        --host localhost \
        --skip-upload
    
    echo -e "${GREEN}✓ Benchmark completed${NC}"
    echo ""
}

# 停止服务
stop_qdrant() {
    echo -e "${YELLOW}[4/5] Stopping Qdrant...${NC}"
    pkill -f "qdrant" || true
    echo -e "${GREEN}✓ Qdrant stopped${NC}"
    echo ""
}

# 显示结果
show_results() {
    echo -e "${YELLOW}[5/5] Benchmark Results${NC}"
    echo ""
    
    LATEST_RESULT=$(ls -t /src/db/vector-db-benchmark/results/*.json 2>/dev/null | head -1)
    
    if [ -n "$LATEST_RESULT" ]; then
        echo "Latest result file: $LATEST_RESULT"
        echo ""
        echo "Summary:"
        cat "$LATEST_RESULT" | python3 -m json.tool | grep -A 5 "mean_time\|mean_precisions\|rps"
    else
        echo "No results found"
    fi
    
    echo ""
    echo -e "${GREEN}=== Benchmark Complete ===${NC}"
}

# 主流程
main() {
    check_environment
    start_qdrant
    
    # 捕获退出信号以确保清理
    trap stop_qdrant EXIT
    
    run_benchmark "$@"
    show_results
}

# 帮助信息
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [DATASET] [CONFIG]"
    echo ""
    echo "Available datasets (examples):"
    echo "  - glove-100-angular (small, fast)"
    echo "  - dbpedia-openai-100K-1536-angular (medium)"
    echo "  - laion-small-clip (large)"
    echo ""
    echo "Available VDE configs:"
    echo "  - qdrant-vde-default"
    echo "  - qdrant-vde-m16-ef100"
    echo "  - qdrant-vde-m32-ef256"
    echo "  - qdrant-vde-vs-native-comparison"
    echo ""
    echo "Example:"
    echo "  $0 glove-100-angular qdrant-vde-default"
    exit 0
fi

# 运行
main "$@"
