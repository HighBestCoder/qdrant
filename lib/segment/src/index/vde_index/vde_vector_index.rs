use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset, TelemetryDetail};
use vde_sys::*;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorRef, VectorInternal};
use crate::index::VectorIndex;
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Distance, Filter, SearchParams};

/// VDE-backed vector index implementation
///
/// This wraps the VDE C++ engine and implements Qdrant's VectorIndex trait.
/// VDE manages both the HNSW index (via vsag) and vector storage (via Btrieve2).
pub struct VDEVectorIndex {
    /// VDE Collection handle (owned by VDE Engine)
    collection: VDECollectionHandle,
    
    /// VDE Engine handle (shared ownership)
    engine: Arc<RwLock<VDEEngineHandle>>,
    
    /// Collection name
    name: String,
    
    /// Vector dimension
    dimension: usize,
    
    /// Distance metric
    #[allow(dead_code)]
    distance: Distance,
    
    /// Base path for files
    path: PathBuf,
}

impl VDEVectorIndex {
    /// Create a new VDE vector index
    pub fn new(
        path: &Path,
        name: &str,
        dimension: usize,
        distance: Distance,
        config_json: Option<&str>,
    ) -> OperationResult<Self> {
        unsafe {
            // Create VDE engine
            let work_dir = std::ffi::CString::new(path.to_str().unwrap())
                .map_err(|e| OperationError::service_error(format!("Invalid path: {}", e)))?;
            
            let engine_handle = vde_engine_create(work_dir.as_ptr());
            if engine_handle.is_null() {
                return Err(OperationError::service_error("Failed to create VDE engine"));
            }
            
            let engine = Arc::new(RwLock::new(engine_handle));
            
            // Prepare collection config
            let collection_name = std::ffi::CString::new(name)
                .map_err(|e| OperationError::service_error(format!("Invalid name: {}", e)))?;
            
            let index_type = std::ffi::CString::new("vsag_hnsw").unwrap();
            let storage_type = std::ffi::CString::new("zendb").unwrap();
            let distance_str = std::ffi::CString::new(match distance {
                Distance::Cosine => "cosine",
                Distance::Euclid => "euclidean",
                Distance::Dot => "dot",
                Distance::Manhattan => "manhattan",
            }).unwrap();
            
            let config_json_cstr = config_json.map(|s| std::ffi::CString::new(s).unwrap());
            
            let config = VDECollectionConfig {
                index_type: index_type.as_ptr(),
                storage_type: storage_type.as_ptr(),
                dimension: dimension as u32,
                distance_metric: distance_str.as_ptr(),
                config_json: config_json_cstr.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            };
            
            // Create collection
            let collection = vde_collection_create(engine_handle, collection_name.as_ptr(), &config);
            if collection.is_null() {
                vde_engine_destroy(engine_handle);
                return Err(OperationError::service_error("Failed to create VDE collection"));
            }
            
            Ok(Self {
                collection,
                engine,
                name: name.to_string(),
                dimension,
                distance,
                path: path.to_path_buf(),
            })
        }
    }
    
    /// Open an existing VDE collection
    pub fn open(
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
            
            let collection = vde_collection_open(engine_handle, collection_name.as_ptr());
            if collection.is_null() {
                vde_engine_destroy(engine_handle);
                return Err(OperationError::service_error("Failed to open VDE collection"));
            }
            
            Ok(Self {
                collection,
                engine,
                name: name.to_string(),
                dimension,
                distance,
                path: path.to_path_buf(),
            })
        }
    }
    
    /// Save index snapshot
    pub fn save(&self) -> OperationResult<()> {
        unsafe {
            let ret = vde_save_snapshot(self.collection);
            if ret != 0 {
                return Err(OperationError::service_error(format!("Failed to save VDE snapshot: {}", ret)));
            }
            Ok(())
        }
    }
}

impl VectorIndex for VDEVectorIndex {
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        _params: Option<&SearchParams>,
        _query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        let mut all_results = Vec::with_capacity(vectors.len());
        
        for query_vector in vectors {
            // Extract dense vector
            let dense = match query_vector {
                QueryVector::Nearest(VectorInternal::Dense(v)) => v.as_slice(),
                _ => return Err(OperationError::service_error("VDE only supports dense vectors")),
            };
            
            if dense.len() != self.dimension {
                return Err(OperationError::service_error(
                    format!("Vector dimension mismatch: expected {}, got {}", self.dimension, dense.len())
                ));
            }
            
            unsafe {
                let vde_query = VDEVector {
                    data: dense.as_ptr() as *mut f32,
                    dim: dense.len() as u32,
                };
                
                let mut results = vec![VDESearchResult { offset: 0, score: 0.0 }; top];
                let mut result_count: u32 = 0;
                
                let ret = if let Some(filter) = filter {
                    // Convert filter to JSON
                    let filter_json = serde_json::to_string(filter)
                        .map_err(|e| OperationError::service_error(format!("Failed to serialize filter: {}", e)))?;
                    let filter_cstr = std::ffi::CString::new(filter_json).unwrap();
                    
                    vde_search_filtered(
                        self.collection,
                        &vde_query,
                        top as u32,
                        filter_cstr.as_ptr(),
                        results.as_mut_ptr(),
                        &mut result_count,
                    )
                } else {
                    vde_search(
                        self.collection,
                        &vde_query,
                        top as u32,
                        results.as_mut_ptr(),
                        &mut result_count,
                    )
                };
                
                if ret != 0 {
                    return Err(OperationError::service_error(format!("VDE search failed: {}", ret)));
                }
                
                // Convert to Qdrant format
                let scored_points: Vec<ScoredPointOffset> = results[..result_count as usize]
                    .iter()
                    .map(|r| ScoredPointOffset {
                        idx: r.offset as PointOffsetType,
                        score: r.score,
                    })
                    .collect();
                
                all_results.push(scored_points);
            }
        }
        
        Ok(all_results)
    }
    
    fn get_telemetry_data(&self, _detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        VectorIndexSearchesTelemetry {
            index_name: Some("vde_hnsw".to_string()),
            unfiltered_plain: Default::default(),
            filtered_plain: Default::default(),
            unfiltered_hnsw: Default::default(),
            filtered_small_cardinality: Default::default(),
            filtered_large_cardinality: Default::default(),
            filtered_exact: Default::default(),
            filtered_sparse: Default::default(),
            unfiltered_sparse: Default::default(),
            unfiltered_exact: Default::default(),
        }
    }
    
    fn files(&self) -> Vec<PathBuf> {
        vec![
            self.path.join(format!("{}.vde", self.name)),
        ]
    }
    
    fn indexed_vector_count(&self) -> usize {
        unsafe {
            vde_get_vector_count(self.collection) as usize
        }
    }
    
    fn size_of_searchable_vectors_in_bytes(&self) -> usize {
        // Estimate: num_vectors * dimension * sizeof(f32) + HNSW overhead
        let vector_count = self.indexed_vector_count();
        vector_count * self.dimension * 4 * 2 // 2x for HNSW graph overhead
    }
    
    fn update_vector(
        &mut self,
        id: PointOffsetType,
        vector: Option<VectorRef>,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        unsafe {
            if let Some(vec_ref) = vector {
                // Insert/update vector
                let dense = match vec_ref {
                    VectorRef::Dense(v) => v,
                    _ => return Err(OperationError::service_error("VDE only supports dense vectors")),
                };
                
                let vde_vector = VDEVector {
                    data: dense.as_ptr() as *mut f32,
                    dim: dense.len() as u32,
                };
                
                let ret = vde_upsert_vector(
                    self.collection,
                    id as u64,
                    &vde_vector,
                    std::ptr::null(), // payload managed separately
                );
                
                if ret != 0 {
                    return Err(OperationError::service_error(format!("Failed to upsert vector: {}", ret)));
                }
            } else {
                // Delete vector
                let ret = vde_delete_vector(self.collection, id as u64);
                if ret != 0 {
                    return Err(OperationError::service_error(format!("Failed to delete vector: {}", ret)));
                }
            }
            
            Ok(())
        }
    }
}

impl Drop for VDEVectorIndex {
    fn drop(&mut self) {
        // Save snapshot before dropping
        let _ = self.save();
        
        // Engine will be destroyed when Arc refcount reaches 0
        if let Ok(engine) = self.engine.read() {
            unsafe {
                if !(*engine).is_null() {
                    vde_engine_destroy(*engine);
                }
            }
        }
    }
}

unsafe impl Send for VDEVectorIndex {}
unsafe impl Sync for VDEVectorIndex {}
