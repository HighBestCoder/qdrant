#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

//! Low-level FFI bindings to VDE (Vector Data Engine)
//!
//! This crate provides raw, unsafe Rust bindings to the VDE C API.
//! For safe, idiomatic Rust wrappers, see the vde_index module in segment.

// Include the auto-generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    #[test]
    fn test_engine_create_destroy() {
        unsafe {
            let work_dir = CString::new("/tmp/vde_test").unwrap();
            let engine = vde_engine_create(work_dir.as_ptr());
            assert!(!engine.is_null(), "Failed to create VDE engine");
            
            vde_engine_destroy(engine);
        }
    }

    #[test]
    fn test_collection_create() {
        unsafe {
            let work_dir = CString::new("/tmp/vde_test_collection").unwrap();
            let engine = vde_engine_create(work_dir.as_ptr());
            assert!(!engine.is_null());

            let name = CString::new("test_collection").unwrap();
            let config = VDECollectionConfig {
                index_type: CString::new("vsag_hnsw").unwrap().as_ptr(),
                storage_type: CString::new("zendb").unwrap().as_ptr(),
                dimension: 128,
                distance_metric: CString::new("cosine").unwrap().as_ptr(),
                config_json: ptr::null(),
            };

            let collection = vde_collection_create(engine, name.as_ptr(), &config);
            assert!(!collection.is_null(), "Failed to create collection");

            vde_engine_destroy(engine);
        }
    }
}
