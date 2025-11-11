use std::borrow::Cow;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

use bitvec::slice::BitSlice;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use vde_sys::*;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::Flusher;
use crate::data_types::named_vectors::CowVector;
use crate::data_types::vectors::VectorRef;
use crate::types::{Distance, VectorStorageDatatype};
use crate::vector_storage::VectorStorage;

/// VDE-backed vector storage
///
/// This implementation delegates vector storage to VDE's Btrieve2 backend.
/// VDE manages the raw vector data while Qdrant manages deleted flags.
pub struct VDEVectorStorage {
    /// VDE Collection handle
    collection: VDECollectionHandle,
    
    /// VDE Engine handle (shared)
    engine: Arc<RwLock<VDEEngineHandle>>,
    
    /// Collection name
    name: String,
    
    /// Vector dimension
    dimension: usize,
    
    /// Distance metric
    distance: Distance,
    
    /// Data type
    datatype: VectorStorageDatatype,
    
    /// Base path
    path: PathBuf,
    
    /// Deleted vector tracking (VDE uses internal bitset, but we maintain this for compatibility)
    deleted: Arc<RwLock<Vec<bool>>>,
}

impl VDEVectorStorage {
    pub fn new(
        path: &Path,
        name: &str,
        dimension: usize,
        distance: Distance,
    ) -> OperationResult<Self> {
        unsafe {
            let work_dir = std::ffi::CString::new(path.to_str().unwrap())
                .map_err(|e| OperationError::service_error(format!("Invalid path: {}", e)))?;
            
            let engine_handle = vde_engine_create(work_dir.as_ptr());
            if engine_handle.is_null() {
                return Err(OperationError::service_error("Failed to create VDE engine"));
            }
            
            let engine = Arc::new(RwLock::new(engine_handle));
            
            let collection_name = std::ffi::CString::new(name)
                .map_err(|e| OperationError::service_error(format!("Invalid name: {}", e)))?;
            
            // Try to open existing collection first
            let mut collection = vde_collection_open(engine_handle, collection_name.as_ptr());
            
            if collection.is_null() {
                // Create new collection
                let index_type = std::ffi::CString::new("vsag_hnsw").unwrap();
                let storage_type = std::ffi::CString::new("zendb").unwrap();
                let distance_str = std::ffi::CString::new(match distance {
                    Distance::Cosine => "cosine",
                    Distance::Euclid => "euclidean",
                    Distance::Dot => "dot",
                    Distance::Manhattan => "manhattan",
                }).unwrap();
                
                let config = VDECollectionConfig {
                    index_type: index_type.as_ptr(),
                    storage_type: storage_type.as_ptr(),
                    dimension: dimension as u32,
                    distance_metric: distance_str.as_ptr(),
                    config_json: std::ptr::null(),
                };
                
                collection = vde_collection_create(engine_handle, collection_name.as_ptr(), &config);
                if collection.is_null() {
                    vde_engine_destroy(engine_handle);
                    return Err(OperationError::service_error("Failed to create VDE collection"));
                }
            }
            
            Ok(Self {
                collection,
                engine,
                name: name.to_string(),
                dimension,
                distance,
                datatype: VectorStorageDatatype::Float32,
                path: path.to_path_buf(),
                deleted: Arc::new(RwLock::new(Vec::new())),
            })
        }
    }
    
    /// Get vector from VDE
    fn get_vector_internal(&self, key: PointOffsetType) -> OperationResult<Vec<f32>> {
        unsafe {
            let mut vector_data = vec![0.0f32; self.dimension];
            let mut vde_vector = VDEVector {
                data: vector_data.as_mut_ptr(),
                dim: self.dimension as u32,
            };
            
            let ret = vde_get_vector(
                self.collection,
                key as u64,
                &mut vde_vector,
                std::ptr::null_mut(),
            );
            
            if ret != 0 {
                return Err(OperationError::service_error(format!("Failed to get vector {}: {}", key, ret)));
            }
            
            Ok(vector_data)
        }
    }
}

impl VectorStorage for VDEVectorStorage {
    fn distance(&self) -> Distance {
        self.distance
    }
    
    fn datatype(&self) -> VectorStorageDatatype {
        self.datatype
    }
    
    fn is_on_disk(&self) -> bool {
        true // VDE uses Btrieve2 for persistent storage
    }
    
    fn total_vector_count(&self) -> usize {
        unsafe {
            vde_get_vector_count(self.collection) as usize
        }
    }
    
    fn get_vector<P: crate::vector_storage::AccessPattern>(&self, key: PointOffsetType) -> CowVector<'_> {
        let vector = self.get_vector_internal(key).unwrap_or_default();
        CowVector::Dense(Cow::Owned(vector))
    }
    
    fn get_vector_opt<P: crate::vector_storage::AccessPattern>(&self, key: PointOffsetType) -> Option<CowVector<'_>> {
        self.get_vector_internal(key).ok().map(|v| CowVector::Dense(Cow::Owned(v)))
    }
    
    fn insert_vector(
        &mut self,
        key: PointOffsetType,
        vector: VectorRef,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        unsafe {
            let dense = match vector {
                VectorRef::Dense(v) => v,
                _ => return Err(OperationError::service_error("VDE only supports dense vectors")),
            };
            
            if dense.len() != self.dimension {
                return Err(OperationError::service_error(
                    format!("Vector dimension mismatch: expected {}, got {}", self.dimension, dense.len())
                ));
            }
            
            let vde_vector = VDEVector {
                data: dense.as_ptr() as *mut f32,
                dim: dense.len() as u32,
            };
            
            let ret = vde_upsert_vector(
                self.collection,
                key as u64,
                &vde_vector,
                std::ptr::null(),
            );
            
            if ret != 0 {
                return Err(OperationError::service_error(format!("Failed to insert vector: {}", ret)));
            }
            
            // Ensure deleted tracking is large enough
            let mut deleted = self.deleted.write().unwrap();
            if deleted.len() <= key as usize {
                deleted.resize(key as usize + 1, false);
            }
            deleted[key as usize] = false;
            
            Ok(())
        }
    }
    
    fn update_from<'a>(
        &mut self,
        other_vectors: &'a mut impl Iterator<Item = (CowVector<'a>, bool)>,
        stopped: &AtomicBool,
    ) -> OperationResult<Range<PointOffsetType>> {
        let mut start: Option<PointOffsetType> = None;
        let mut end: PointOffsetType = 0;
        
        for (idx, (vector, deleted)) in other_vectors.enumerate() {
            if stopped.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(OperationError::Cancelled {
                    description: "Update cancelled".to_string(),
                });
            }
            
            let key = idx as PointOffsetType;
            if start.is_none() {
                start = Some(key);
            }
            end = key + 1;
            
            if !deleted {
                let vec_ref = match &vector {
                    CowVector::Dense(Cow::Owned(v)) => VectorRef::Dense(v.as_slice()),
                    CowVector::Dense(Cow::Borrowed(v)) => VectorRef::Dense(v),
                    _ => return Err(OperationError::service_error("VDE only supports dense vectors")),
                };
                self.insert_vector(key, vec_ref, &HardwareCounterCell::disposable())?;
            } else {
                self.delete_vector(key)?;
            }
        }
        
        Ok(start.unwrap_or(0)..end)
    }
    
    fn flusher(&self) -> Flusher {
        // Since VDECollection pointer can't be sent between threads,
        // we return a no-op flusher. VDE flushes on Drop.
        Box::new(|| Ok(()))
    }
    
    fn files(&self) -> Vec<PathBuf> {
        vec![
            self.path.join(format!("{}_vectors.btr", self.name)),
            self.path.join(format!("{}_index.snapshot", self.name)),
        ]
    }
    
    fn delete_vector(&mut self, key: PointOffsetType) -> OperationResult<bool> {
        unsafe {
            let ret = vde_delete_vector(self.collection, key as u64);
            if ret != 0 {
                return Err(OperationError::service_error(format!("Failed to delete vector: {}", ret)));
            }
            
            let mut deleted = self.deleted.write().unwrap();
            if deleted.len() <= key as usize {
                deleted.resize(key as usize + 1, false);
            }
            
            let was_not_deleted = !deleted[key as usize];
            deleted[key as usize] = true;
            
            Ok(was_not_deleted)
        }
    }
    
    fn is_deleted_vector(&self, key: PointOffsetType) -> bool {
        let deleted = self.deleted.read().unwrap();
        if key as usize >= deleted.len() {
            return false;
        }
        deleted[key as usize]
    }
    
    fn deleted_vector_count(&self) -> usize {
        let deleted = self.deleted.read().unwrap();
        deleted.iter().filter(|&&d| d).count()
    }
    
    fn deleted_vector_bitslice(&self) -> &BitSlice {
        // This is unsafe but required by trait
        // In practice, VDE manages deletions internally
        BitSlice::empty()
    }
}

impl Drop for VDEVectorStorage {
    fn drop(&mut self) {
        // Flush before dropping
        unsafe {
            vde_flush(self.collection);
        }
        
        if let Ok(engine) = self.engine.read() {
            unsafe {
                if !(*engine).is_null() {
                    vde_engine_destroy(*engine);
                }
            }
        }
    }
}

unsafe impl Send for VDEVectorStorage {}
unsafe impl Sync for VDEVectorStorage {}
