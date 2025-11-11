# VDE 性能测试指南

使用 vector-db-benchmark 对 Qdrant + VDE 进行性能测试。

## 快速开始

### 1. 启动 Qdrant with VDE

```bash
cd /src/db/qdrant
./run_qdrant_with_vde.sh ./target/release/qdrant
```

验证服务运行：
```bash
curl http://localhost:6333/healthz
```

### 2. 运行基础性能测试

使用提供的自动化脚本：

```bash
cd /src/db/qdrant
./run_vde_benchmark.sh
```

这将：
- 自动启动 Qdrant VDE 服务器
- 运行默认的性能测试
- 生成测试报告
- 清理环境

### 3. 运行对比测试 (VDE vs 原生 HNSW)

```bash
cd /src/db/qdrant
./compare_vde_vs_native.sh glove-100-angular 8
```

参数说明：
- `glove-100-angular`: 数据集名称
- `8`: 并发请求数

## 测试配置

### 可用的 VDE 配置

1. **qdrant-vde-default** - 默认配置
   - M: 16
   - ef_construct: 100
   - 适合快速测试

2. **qdrant-vde-m16-ef100** - 多参数测试
   - 测试不同的 hnsw_ef 值 (64, 128, 256)
   - 测试不同的并发级别 (1, 8)

3. **qdrant-vde-m32-ef256** - 高质量配置
   - M: 32
   - ef_construct: 256
   - 更高的召回率

4. **qdrant-vde-vs-native-comparison** - 对比测试
   - 与原生 HNSW 相同参数
   - 用于直接性能对比

### 可用数据集

| 数据集 | 向量数 | 维度 | 距离 | 大小 |
|--------|--------|------|------|------|
| glove-100-angular | 1.2M | 100 | Angular | 小 |
| dbpedia-openai-100K-1536-angular | 100K | 1536 | Angular | 中 |
| laion-small-clip | 10M | 512 | Cosine | 大 |

## 手动测试步骤

### 1. 准备环境

安装 vector-db-benchmark 依赖：

```bash
cd /src/db/vector-db-benchmark
pip install poetry
poetry install
```

### 2. 启动 Qdrant 服务器

```bash
# 设置环境变量
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export LD_LIBRARY_PATH="/src/db/vde/build/lib:/usr/local/actianzen/lib64:$LD_LIBRARY_PATH"

# 启动服务
cd /src/db/qdrant
./target/release/qdrant
```

### 3. 运行测试

```bash
cd /src/db/vector-db-benchmark
poetry shell

# 测试单个配置
python run.py \
    --engines "qdrant-vde-default" \
    --datasets "glove-100-angular" \
    --host localhost

# 测试多个配置
python run.py \
    --engines "qdrant-vde-*" \
    --datasets "glove-*" \
    --host localhost
```

### 4. 查看结果

结果保存在 `./results/` 目录：

```bash
ls -lth ./results/

# 查看最新结果
cat ./results/$(ls -t ./results/*.json | head -1) | python -m json.tool
```

## 性能指标说明

测试会输出以下关键指标：

- **mean_time**: 平均查询时间（秒）
- **rps**: 每秒请求数（Requests Per Second）
- **mean_precisions**: 平均召回率（@ k）
- **p95_time**: 95分位延迟
- **p99_time**: 99分位延迟

## 自定义测试配置

编辑配置文件：

```bash
vim /src/db/vector-db-benchmark/experiments/configurations/qdrant-vde.json
```

配置格式：

```json
{
  "name": "my-vde-test",
  "engine": "qdrant",
  "collection_params": {
    "vectors": {
      "storage_type": "vde"
    },
    "hnsw_config": {
      "type": "vde_hnsw",
      "m": 16,
      "ef_construct": 100
    }
  },
  "search_params": [
    { "parallel": 8, "config": { "hnsw_ef": 128 } }
  ],
  "upload_params": { "parallel": 16, "batch_size": 1024 }
}
```

## 测试场景示例

### 场景 1: 延迟优化测试

测试不同 `hnsw_ef` 对延迟的影响：

```bash
python run.py \
    --engines "qdrant-vde-m16-ef100" \
    --datasets "glove-100-angular" \
    --host localhost
```

### 场景 2: 吞吐量测试

测试不同并发级别：

```json
"search_params": [
  { "parallel": 1, "config": { "hnsw_ef": 128 } },
  { "parallel": 4, "config": { "hnsw_ef": 128 } },
  { "parallel": 8, "config": { "hnsw_ef": 128 } },
  { "parallel": 16, "config": { "hnsw_ef": 128 } }
]
```

### 场景 3: 召回率 vs 性能

测试精度和性能的权衡：

```json
"search_params": [
  { "parallel": 8, "config": { "hnsw_ef": 64 } },   // 快但精度低
  { "parallel": 8, "config": { "hnsw_ef": 128 } },  // 平衡
  { "parallel": 8, "config": { "hnsw_ef": 256 } }   // 慢但精度高
]
```

## 结果分析

### 生成可视化报告

```bash
cd /src/db/vector-db-benchmark

# 生成图表（需要安装 matplotlib）
poetry run python tools/plot_results.py --results ./results/*.json
```

### 对比分析

使用 jq 对比两个测试结果：

```bash
# VDE 结果
jq '.results[] | {config: .name, rps, recall: .mean_precisions}' \
    results/qdrant-vde-default-*.json

# 原生 HNSW 结果
jq '.results[] | {config: .name, rps, recall: .mean_precisions}' \
    results/qdrant-m-16-ef-100-*.json
```

## 常见问题

### Q: 测试失败，显示连接超时

**A**: 确保 Qdrant 服务正在运行：
```bash
curl http://localhost:6333/healthz
ps aux | grep qdrant
```

### Q: 找不到数据集

**A**: 首次运行会自动下载数据集到 `~/.cache/dbsz/`，需要网络连接和足够磁盘空间。

### Q: VDE 配置不生效

**A**: 确保：
1. Qdrant 使用了 VDE 构建版本
2. 配置文件中正确设置了 `"storage_type": "vde"`
3. 检查 Qdrant 日志确认 VDE 模块已加载

### Q: 性能比原生差

**A**: 可能的原因：
1. VDE 配置参数未优化（尝试调整 M 和 ef_construct）
2. Btrieve 数据库文件 I/O 瓶颈
3. 测试数据集规模太小，无法体现 VDE 优势

## 进阶：大规模测试

### 使用大数据集

```bash
# 下载并测试 10M 向量
python run.py \
    --engines "qdrant-vde-m32-ef256" \
    --datasets "laion-small-clip" \
    --host localhost
```

### 分布式测试

修改配置使用多个 Qdrant 节点（需要集群配置）。

### 持续性能监控

使用 Prometheus + Grafana 监控 Qdrant 指标：

```bash
# Qdrant 暴露 metrics
curl http://localhost:6333/metrics
```

## 提交结果

将测试结果上传到 Qdrant 官方 benchmark 网站：

```bash
python run.py \
    --engines "qdrant-vde-*" \
    --datasets "glove-*" \
    --host localhost \
    --no-skip-upload
```

## 总结

性能测试的关键步骤：

1. ✅ 构建 Release 版本的 Qdrant
2. ✅ 配置 VDE 参数
3. ✅ 选择合适的数据集
4. ✅ 运行基准测试
5. ✅ 分析结果并优化
6. ✅ 对比 VDE vs 原生性能

Happy Benchmarking! 🚀
