# VDE 与 Qdrant 集成流程详解

## 核心流程概览

```
VDE (C++) ──编译──> libvde.a ──链接──> vde-sys (Rust FFI) ──依赖──> segment ──> Qdrant
```

---

## 1. VDE 编译阶段

### 1.1 编译 VDE 静态库

```bash
cd /src/db/vde
mkdir -p build && cd build
cmake ..
make -j16
```

**输出产物**：
- `/src/db/vde/build/lib/libvde.a` - VDE 静态库
- `/src/db/vde/build/lib/libvsag.a` - VSAG 索引库（VDE 依赖）

**libvde.a 包含的模块**：
```cpp
// 来自 vde/src/CMakeLists.txt
VDE_SOURCES = {
    vde/vde_engine.cpp
    vde/vde_collection.cpp
    vde/vde_c_api.cpp              // ← C API 接口
    index_manager/index_manager.cpp
    index_manager/index_drivers/vsag_driver.cpp
    storage_manager/storage_manager.cpp
    storage_manager/storage_drivers/zendb_driver.cpp
    storage_manager/storage_drivers/mem_driver.cpp   // ← Memory Driver
}
```

---

## 2. Rust FFI 绑定层 (vde-sys)

### 2.1 目录结构

```
qdrant/lib/vde-sys/
├── Cargo.toml          # Rust crate 配置
├── build.rs            # 构建脚本（链接 libvde.a）
├── wrapper.h           # C 头文件包装
└── src/
    └── lib.rs          # Rust FFI 绑定
```

### 2.2 build.rs - 核心链接逻辑

**文件**: `/src/db/qdrant/lib/vde-sys/build.rs`

```rust
fn main() {
    // 1. 定位 VDE 源码路径
    let vde_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../vde");
    
    let vde_include = vde_dir.join("include");
    let vde_build = vde_dir.join("build/lib");
    
    // 2. ★ 链接 VDE 静态库 ★
    println!("cargo:rustc-link-search=native={}", vde_build.display());
    println!("cargo:rustc-link-lib=static=vde");
    // 输出: cargo:rustc-link-search=native=/src/db/vde/build/lib
    // 输出: cargo:rustc-link-lib=static=vde
    
    // 3. 链接 VDE 的依赖库
    println!("cargo:rustc-link-lib=dylib=stdc++");     // C++ 标准库
    println!("cargo:rustc-link-lib=dylib=vsag");       // VSAG 索引库
    
    // 4. 链接 Btrieve2 库
    println!("cargo:rustc-link-search=native=/usr/local/actianzen/lib64");
    println!("cargo:rustc-link-lib=dylib=btrieveCpp");
    println!("cargo:rustc-link-lib=dylib=btrieveC");
    
    // 5. 生成 Rust FFI 绑定（使用 bindgen）
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", vde_include.display()))
        .allowlist_type("VDE.*")
        .allowlist_function("vde_.*")
        .generate()
        .expect("Unable to generate bindings");
    
    // 6. 写入绑定文件到 OUT_DIR
    bindings.write_to_file(out_path.join("bindings.rs"));
}
```

**关键点**：
- `cargo:rustc-link-search=native=` - 告诉 Rust 编译器去哪里找库文件
- `cargo:rustc-link-lib=static=vde` - 链接静态库 `libvde.a`
- `cargo:rustc-link-lib=dylib=` - 链接动态库（`.so` 文件）

### 2.3 wrapper.h - C API 包装

**文件**: `/src/db/qdrant/lib/vde-sys/wrapper.h`

```cpp
#include "../../../vde/include/vde/vde.h"
```

这个文件让 `bindgen` 知道要为哪些 C API 生成 Rust 绑定。

### 2.4 lib.rs - 生成的绑定

**文件**: `/src/db/qdrant/lib/vde-sys/src/lib.rs`

```rust
// 包含编译时生成的绑定
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// 这会展开为类似：
pub type VDEEngineHandle = *mut VDEEngine;
pub type VDECollectionHandle = *mut VDECollection;

extern "C" {
    pub fn vde_engine_create(work_dir: *const c_char) -> VDEEngineHandle;
    pub fn vde_collection_create(
        engine: VDEEngineHandle,
        name: *const c_char,
        config: *const VDECollectionConfig
    ) -> VDECollectionHandle;
    // ... 更多 FFI 函数
}
```

---

## 3. Segment 使用 vde-sys

### 3.1 Cargo.toml 依赖声明

**文件**: `/src/db/qdrant/lib/segment/Cargo.toml`

```toml
[dependencies]
vde-sys = { path = "../vde-sys" }
```

### 3.2 VDEVectorStorage 实现

**文件**: `/src/db/qdrant/lib/segment/src/vector_storage/vde_storage/vde_vector_storage.rs`

```rust
use vde_sys::*;  // 导入 FFI 绑定

pub struct VDEVectorStorage {
    collection: VDECollectionHandle,  // C 指针
    engine: Arc<RwLock<VDEEngineHandle>>,
    // ...
}

impl VDEVectorStorage {
    pub fn new(path: &Path, name: &str, dimension: usize) -> Self {
        unsafe {
            // 1. 创建 VDE Engine
            let engine_handle = vde_engine_create(work_dir.as_ptr());
            
            // 2. 配置 Collection（指定 storage driver）
            let storage_type = CString::new("zendb").unwrap();  // 或 "memory"
            let config = VDECollectionConfig {
                index_type: CString::new("vsag_hnsw").unwrap().as_ptr(),
                storage_type: storage_type.as_ptr(),  // ← 指定驱动
                dimension: dimension as u32,
                distance_metric: distance_str.as_ptr(),
                config_json: std::ptr::null(),
            };
            
            // 3. 创建 Collection
            let collection = vde_collection_create(
                engine_handle, 
                collection_name.as_ptr(), 
                &config
            );
            
            Self { collection, engine, ... }
        }
    }
    
    fn insert_vector(&mut self, key: PointOffsetType, vector: VectorRef) {
        unsafe {
            let vde_vector = VDEVector {
                data: dense.as_ptr() as *mut f32,
                dim: dense.len() as u32,
            };
            
            // 调用 C API
            vde_upsert_vector(self.collection, key as u64, &vde_vector, payload);
        }
    }
}
```

---

## 4. 完整编译链路

### 4.1 编译顺序

```bash
# 步骤 1: 编译 VDE C++ 代码
cd /src/db/vde/build
cmake ..
make -j16
# 产出: build/lib/libvde.a

# 步骤 2: 编译 Qdrant (自动触发 vde-sys 构建)
cd /src/db/qdrant
cargo build --release

# vde-sys build.rs 执行流程:
# 1. 查找 /src/db/vde/build/lib/libvde.a
# 2. 生成 Rust FFI 绑定 (bindings.rs)
# 3. 链接 libvde.a、libstdc++、libvsag、libbtrieveCpp

# 步骤 3: segment crate 依赖 vde-sys
# 自动链接所有依赖库
```

### 4.2 链接器命令（简化版）

```bash
# Rust 编译器最终生成的链接器命令类似：
rustc \
  -L /src/db/vde/build/lib \
  -L /usr/local/actianzen/lib64 \
  -l static=vde \
  -l dylib=stdc++ \
  -l dylib=vsag \
  -l dylib=btrieveCpp \
  -l dylib=btrieveC \
  --crate-name segment \
  lib/segment/src/lib.rs
```

---

## 5. 依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                  Qdrant Binary                          │
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │         segment (Rust crate)                  │    │
│  │                                               │    │
│  │  ┌─────────────────────────────────────┐     │    │
│  │  │    vde-sys (Rust FFI crate)         │     │    │
│  │  │                                      │     │    │
│  │  │  ┌────────────────────────────┐     │     │    │
│  │  │  │   libvde.a (C++ static)    │     │     │    │
│  │  │  │                            │     │     │    │
│  │  │  │  - vde_engine.cpp          │     │     │    │
│  │  │  │  - vde_collection.cpp      │     │     │    │
│  │  │  │  - vde_c_api.cpp ◄─────────┼─────┼─────┼─── FFI 调用
│  │  │  │  - storage_manager.cpp     │     │     │    │
│  │  │  │  - zendb_driver.cpp        │     │     │    │
│  │  │  │  - mem_driver.cpp          │     │     │    │
│  │  │  │  - vsag_driver.cpp         │     │     │    │
│  │  │  └────────────────────────────┘     │     │    │
│  │  │                ▲                     │     │    │
│  │  │                │ links               │     │    │
│  │  │                ▼                     │     │    │
│  │  │  ┌────────────────────────────┐     │     │    │
│  │  │  │   libvsag.so (动态库)      │     │     │    │
│  │  │  └────────────────────────────┘     │     │    │
│  │  │                                      │     │    │
│  │  │  ┌────────────────────────────┐     │     │    │
│  │  │  │   libbtrieveCpp.so         │     │     │    │
│  │  │  │   libbtrieveC.so           │     │     │    │
│  │  │  └────────────────────────────┘     │     │    │
│  │  │                                      │     │    │
│  │  │  ┌────────────────────────────┐     │     │    │
│  │  │  │   libstdc++.so             │     │     │    │
│  │  │  └────────────────────────────┘     │     │    │
│  │  └─────────────────────────────────────┘     │    │
│  └───────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 切换 Storage Driver

### 6.1 使用 Memory Driver

修改 `vde_vector_storage.rs`:

```rust
// 第 103 行
let storage_type = std::ffi::CString::new("memory").unwrap();  // 改为 "memory"

let config = VDECollectionConfig {
    storage_type: storage_type.as_ptr(),  // 使用内存驱动
    // ...
};
```

### 6.2 使用 ZenDB Driver

```rust
let storage_type = std::ffi::CString::new("zendb").unwrap();  // 使用 "zendb"
```

### 6.3 驱动注册表

**文件**: `/src/db/vde/src/storage_manager/storage_manager.cpp`

```cpp
void RegisterBuiltinDrivers() {
    RegisterDriver("zendb", CreateZenDBDriver);
    RegisterDriver("btrieve2", CreateZenDBDriver);    // 别名
    
    RegisterDriver("memory", CreateMemDriver);
    RegisterDriver("mem", CreateMemDriver);           // 别名
}
```

---

## 7. 常见问题排查

### 7.1 链接错误：找不到 libvde.a

```
error: linking with `cc` failed
  = note: /usr/bin/ld: cannot find -lvde
```

**原因**：VDE 没有编译，或 build.rs 路径不对

**解决**：
```bash
cd /src/db/vde/build
make -j16
ls -lh lib/libvde.a  # 确认存在
```

### 7.2 链接错误：undefined reference to `CreateMemDriver`

```
undefined reference to `vde::CreateMemDriver(vde::StorageConfig const&)`
```

**原因**：`mem_driver.cpp` 没有加入 CMakeLists.txt

**解决**：
检查 `/src/db/vde/src/CMakeLists.txt`:
```cmake
set(VDE_SOURCES
    # ...
    storage_manager/storage_drivers/mem_driver.cpp  # ← 必须存在
)
```

### 7.3 运行时错误：找不到 libbtrieveCpp.so

```
error while loading shared libraries: libbtrieveCpp.so: cannot open shared object file
```

**原因**：动态库路径未设置

**解决**：
```bash
export LD_LIBRARY_PATH=/usr/local/actianzen/lib64:$LD_LIBRARY_PATH
```

或在启动脚本中设置：
```bash
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/actianzen/lib64:$LD_LIBRARY_PATH
./target/release/qdrant
```

---

## 8. 开发工作流

### 8.1 修改 VDE C++ 代码后

```bash
# 1. 重新编译 VDE
cd /src/db/vde/build
make -j16

# 2. 重新编译 Qdrant（会自动链接新的 libvde.a）
cd /src/db/qdrant
cargo build

# 3. 如果修改了 C API（vde.h），需要重新生成绑定
cd lib/vde-sys
cargo clean
cargo build
```

### 8.2 添加新的 Storage Driver

```bash
# 1. 创建 driver 文件
/src/db/vde/src/storage_manager/storage_drivers/new_driver.cpp
/src/db/vde/src/storage_manager/storage_drivers/new_driver.h

# 2. 修改 CMakeLists.txt
vim /src/db/vde/src/CMakeLists.txt
# 添加: storage_manager/storage_drivers/new_driver.cpp

# 3. 注册 driver
vim /src/db/vde/src/storage_manager/storage_manager.cpp
# RegisterDriver("newdriver", CreateNewDriver);

# 4. 重新编译
cd /src/db/vde/build && make -j16
cd /src/db/qdrant && cargo build
```

---

## 9. 总结

**VDE → Qdrant 集成的三个关键点**：

1. ✅ **静态链接**: `libvde.a` 通过 `cargo:rustc-link-lib=static=vde` 链接到 Rust
2. ✅ **FFI 绑定**: `bindgen` 自动生成 Rust 绑定
3. ✅ **依赖传递**: 
   - `vde-sys` → 链接 `libvde.a + 依赖库`
   - `segment` → 依赖 `vde-sys`
   - `qdrant` → 依赖 `segment`

**性能优势**：
- 零拷贝 FFI 调用
- 静态链接减少运行时开销
- C++ 和 Rust 代码紧密集成
