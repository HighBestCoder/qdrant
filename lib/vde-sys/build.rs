use std::env;
use std::path::PathBuf;

fn main() {
    // VDE 源码路径
    let vde_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../vde");
    
    let vde_include = vde_dir.join("include");
    let vde_build = vde_dir.join("build/lib");
    
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", vde_include.display());
    
    // 链接 VDE 静态库
    println!("cargo:rustc-link-search=native={}", vde_build.display());
    println!("cargo:rustc-link-lib=static=vde");
    
    // 链接 VDE 依赖的库
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=vsag");
    
    // 链接 Btrieve2 - 使用实际安装路径
    println!("cargo:rustc-link-search=native=/usr/local/actianzen/lib64");
    println!("cargo:rustc-link-lib=dylib=btrieveCpp");
    println!("cargo:rustc-link-lib=dylib=btrieveC");
    
    // 生成 Rust FFI 绑定
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", vde_include.display()))
        // 只生成 VDE 相关的类型
        .allowlist_type("VDE.*")
        .allowlist_function("vde_.*")
        .allowlist_var("VDE_.*")
        // 不生成布局测试（减少编译警告）
        .layout_tests(false)
        // 使用 core 而不是 std（更好的兼容性）
        .use_core()
        // 为 C 枚举生成 Rust 枚举
        .rustified_enum("VDECollectionState")
        // 生成注释
        .generate_comments(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // 写入绑定文件
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
