帮我看一下qdrant的代码和文档
要特别认真

[1] qdrant支持几种normalize方法。
[2] 客户端是否支持指定normalize方法
[3] 不同的metric_type与normalize方法的映射关系是什么样的？
[4] 如果客户端指定的metric_type=cosine，但是指定的normalie方法不是L2_normalize,qdrant是如何处理的？

---

## 详细回答

### [1] Qdrant支持几种normalize方法

**Qdrant实际上不支持让用户显式指定normalize方法，而是根据Distance类型自动进行预处理。**

从代码中可以看到，Qdrant支持4种距离度量类型（`Distance` enum）：

```rust
// lib/segment/src/types.rs:292
pub enum Distance {
    Cosine,      // 余弦距离
    Euclid,      // 欧几里得距离  
    Dot,         // 点积
    Manhattan,   // 曼哈顿距离
}
```

每种Distance类型对应的**预处理（preprocess）方法**如下：

```rust
// lib/segment/src/types.rs:313-325
pub fn preprocess_vector<T: PrimitiveVectorElement>(&self, vector: DenseVector) -> DenseVector {
    match self {
        Distance::Cosine => CosineMetric::preprocess(vector),     // L2归一化
        Distance::Euclid => EuclidMetric::preprocess(vector),     // 不做预处理
        Distance::Dot => DotProductMetric::preprocess(vector),    // 不做预处理
        Distance::Manhattan => ManhattanMetric::preprocess(vector), // 不做预处理
    }
}
```

**关键点：只有Cosine距离会进行L2归一化（normalize）**

### [2] 客户端是否支持指定normalize方法

**不支持！客户端无法显式指定normalize方法。**

从API定义可以看到，客户端只能指定：

```rust
// lib/collection/src/operations/types.rs:1401-1440
pub struct VectorParams {
    pub size: NonZeroU64,              // 向量维度
    pub distance: Distance,             // 距离类型（唯一可配置的）
    pub hnsw_config: Option<HnswConfigDiff>,
    pub quantization_config: Option<QuantizationConfig>,
    pub on_disk: Option<bool>,
    pub datatype: Option<Datatype>,
    pub multivector_config: Option<MultiVectorConfig>,
}
```

**客户端只能指定`distance`字段，无法指定normalize方法。**

### [3] 不同的metric_type与normalize方法的映射关系

| Distance类型 | Normalize方法 | 实现位置 | 具体行为 |
|-------------|-------------|---------|---------|
| **Cosine** | **L2 Normalize** | `lib/segment/src/spaces/simple.rs:228` | `向量/sqrt(向量的L2范数)` |
| Euclid | 无 | `lib/segment/src/spaces/simple.rs:69` | 直接返回原向量 |
| Dot | 无 | `lib/segment/src/spaces/simple.rs:157` | 直接返回原向量 |
| Manhattan | 无 | `lib/segment/src/spaces/simple.rs:113` | 直接返回原向量 |

**核心代码 - Cosine的L2归一化实现：**

```rust
// lib/segment/src/spaces/simple.rs:228-236
pub fn cosine_preprocess(vector: DenseVector) -> DenseVector {
    let mut length: f32 = vector.iter().map(|x| x * x).sum();  // 计算L2范数的平方
    if is_length_zero_or_normalized(length) {
        return vector;  // 如果已经归一化或为零向量，直接返回
    }
    length = length.sqrt();  // 开方得到L2范数
    vector.iter().map(|x| x / length).collect()  // 每个元素除以L2范数
}
```

**关键判断函数：**

```rust
// lib/segment/src/spaces/simple.rs
fn is_length_zero_or_normalized(length: f32) -> bool {
    const EPSILON: f32 = 1e-6;
    length < EPSILON || (length - 1.0).abs() < EPSILON
}
```

### [4] 如果客户端指定metric_type=cosine，但指定的normalize方法不是L2_normalize，qdrant如何处理？

**这个问题的前提不成立！因为：**

1. **客户端无法指定normalize方法**，只能指定`distance`类型
2. **Qdrant会强制对Cosine距离进行L2归一化**

**完整的向量处理流程：**

```
客户端上传向量
    ↓
API接收 (upsert_point / update_vectors)
    ↓
lib/segment/src/segment/entry.rs:127
    vectors.preprocess(|name| self.config().vector_data.get(name).unwrap())
    ↓
lib/segment/src/data_types/named_vectors.rs:313
    根据每个向量的distance配置调用preprocess
    ↓
lib/types.rs:321 (如果distance=Cosine)
    CosineMetric::preprocess(vector)
    ↓
lib/segment/src/spaces/simple.rs:228
    进行L2归一化: x / ||x||
    ↓
存储到向量存储中
```

**核心代码位置：**

```rust
// lib/segment/src/segment/entry.rs:135
fn upsert_point(
    &mut self,
    op_num: SeqNumberType,
    point_id: PointIdType,
    mut vectors: NamedVectors,
    hw_counter: &HardwareCounterCell,
) -> OperationResult<bool> {
    check_named_vectors(&vectors, &self.segment_config)?;
    // 关键：这里会强制预处理，Cosine会被L2归一化
    vectors.preprocess(|name| self.config().vector_data.get(name).unwrap());
    // ... 后续插入逻辑
}
```

**结论：**
- Qdrant**不允许**用户跳过或自定义normalize方法
- 如果指定`distance=Cosine`，向量**必定会被L2归一化**
- 这是在服务端**强制执行**的，客户端无法绕过
- 即使客户端上传的向量已经归一化，Qdrant仍会再次检查并归一化（通过`is_length_zero_or_normalized`避免重复计算）

**额外优化：**
对于Cosine距离，Qdrant使用了SIMD优化的归一化实现：
- AVX2优化：`cosine_preprocess_avx` 
- SSE优化：`cosine_preprocess_sse`
- NEON优化（ARM）：`cosine_preprocess_neon`

这些实现在 `lib/segment/src/spaces/simple_avx.rs`、`simple_sse.rs`、`simple_neon.rs` 中。