#!/bin/bash
# Qdrant with VDE Runtime Script
# 设置 VDE 相关库的运行时路径

# 强制使用系统的 libstdc++ 以避免版本冲突
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# VDE 库路径
export LD_LIBRARY_PATH="/src/db/vde/build/lib:/usr/local/actianzen/lib64:$LD_LIBRARY_PATH"

# 显示加载的库路径
echo "=== VDE Runtime Environment ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# 检查库是否可用
echo "=== Checking Required Libraries ==="
echo -n "libvsag.so: "
if ldconfig -p | grep -q libvsag.so; then
    echo "✓ Found (system)"
elif [ -f "/src/db/vde/build/lib/libvsag.so" ]; then
    echo "✓ Found (local)"
else
    echo "✗ NOT FOUND"
fi

echo -n "libbtrieveCpp.so: "
if ldconfig -p | grep -q libbtrieveCpp.so; then
    echo "✓ Found (system)"
elif [ -f "/usr/local/actianzen/lib64/libbtrieveCpp.so" ]; then
    echo "✓ Found (local)"
else
    echo "✗ NOT FOUND"
fi

echo ""
echo "=== Starting Qdrant with VDE ==="
echo ""

# 运行 Qdrant
exec "$@"
