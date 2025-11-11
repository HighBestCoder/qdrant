use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use serde_json::Value;
use vde_sys::*;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::Flusher;
use crate::json_path::JsonPath;
use crate::payload_storage::PayloadStorage;
use crate::types::Payload;

/// VDE-backed payload storage
///
/// This implementation stores payload (metadata) in VDE's Btrieve2 backend.
/// Payloads are stored as JSON and indexed for filtering.
pub struct VDEPayloadStorage {
    /// VDE Collection handle
    collection: VDECollectionHandle,
    
    /// VDE Engine handle (shared)
    engine: Arc<RwLock<VDEEngineHandle>>,
    
    /// Collection name
    name: String,
    
    /// Base path
    path: PathBuf,
    
    /// In-memory cache for payloads (optional optimization)
    cache: Arc<RwLock<HashMap<PointOffsetType, Payload>>>,
}

impl VDEPayloadStorage {
    pub fn new(
        path: &Path,
        name: &str,
        dimension: usize,
        distance: crate::types::Distance,
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
            
            // Try to open existing collection
            let mut collection = vde_collection_open(engine_handle, collection_name.as_ptr());
            
            if collection.is_null() {
                // Create new collection
                let index_type = std::ffi::CString::new("vsag_hnsw").unwrap();
                let storage_type = std::ffi::CString::new("zendb").unwrap();
                let distance_str = std::ffi::CString::new(match distance {
                    crate::types::Distance::Cosine => "cosine",
                    crate::types::Distance::Euclid => "euclidean",
                    crate::types::Distance::Dot => "dot",
                    crate::types::Distance::Manhattan => "manhattan",
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
                path: path.to_path_buf(),
                cache: Arc::new(RwLock::new(HashMap::new())),
            })
        }
    }
    
    /// Get payload from VDE
    fn get_payload_internal(&self, point_id: PointOffsetType) -> OperationResult<Payload> {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(payload) = cache.get(&point_id) {
                return Ok(payload.clone());
            }
        }
        
        unsafe {
            // Allocate buffer for payload JSON
            let mut json_buffer = vec![0u8; 64 * 1024]; // 64KB max
            let mut vde_payload = VDEPayload {
                json: json_buffer.as_mut_ptr() as *const i8,
                length: json_buffer.len() as u32,
            };
            
            let ret = vde_get_vector(
                self.collection,
                point_id as u64,
                std::ptr::null_mut(),
                &mut vde_payload,
            );
            
            if ret != 0 {
                return Ok(Payload::default()); // Empty payload if not found
            }
            
            // Parse JSON
            let json_str = std::str::from_utf8(&json_buffer[..vde_payload.length as usize])
                .map_err(|e| OperationError::service_error(format!("Invalid UTF-8 in payload: {}", e)))?;
            
            let payload: Payload = serde_json::from_str(json_str)
                .map_err(|e| OperationError::service_error(format!("Failed to parse payload JSON: {}", e)))?;
            
            // Update cache
            {
                let mut cache = self.cache.write().unwrap();
                cache.insert(point_id, payload.clone());
            }
            
            Ok(payload)
        }
    }
    
    /// Set payload in VDE
    fn set_payload_internal(&mut self, point_id: PointOffsetType, payload: &Payload) -> OperationResult<()> {
        unsafe {
            // Serialize payload to JSON
            let json = serde_json::to_string(payload)
                .map_err(|e| OperationError::service_error(format!("Failed to serialize payload: {}", e)))?;
            
            let json_cstr = std::ffi::CString::new(json)
                .map_err(|e| OperationError::service_error(format!("Invalid JSON string: {}", e)))?;
            
            let vde_payload = VDEPayload {
                json: json_cstr.as_ptr(),
                length: json_cstr.as_bytes().len() as u32,
            };
            
            let ret = vde_upsert_vector(
                self.collection,
                point_id as u64,
                std::ptr::null(), // vector handled separately
                &vde_payload,
            );
            
            if ret != 0 {
                return Err(OperationError::service_error(format!("Failed to set payload: {}", ret)));
            }
            
            // Update cache
            {
                let mut cache = self.cache.write().unwrap();
                cache.insert(point_id, payload.clone());
            }
            
            Ok(())
        }
    }
}

impl PayloadStorage for VDEPayloadStorage {
    fn overwrite(
        &mut self,
        point_id: PointOffsetType,
        payload: &Payload,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        self.set_payload_internal(point_id, payload)
    }
    
    fn set(
        &mut self,
        point_id: PointOffsetType,
        payload: &Payload,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        // Get existing payload and merge
        let mut existing = self.get_payload_internal(point_id)?;
        
        // Merge payloads
        existing.merge(payload);
        
        self.set_payload_internal(point_id, &existing)
    }
    
    fn set_by_key(
        &mut self,
        point_id: PointOffsetType,
        payload: &Payload,
        key: &JsonPath,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        let mut existing = self.get_payload_internal(point_id)?;
        
        // Merge by key using JsonPath
        existing.merge_by_key(payload, key);
        
        self.set_payload_internal(point_id, &existing)
    }
    
    fn get(
        &self,
        point_id: PointOffsetType,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Payload> {
        self.get_payload_internal(point_id)
    }
    
    fn get_sequential(
        &self,
        point_id: PointOffsetType,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Payload> {
        self.get(point_id, hw_counter)
    }
    
    fn delete(
        &mut self,
        point_id: PointOffsetType,
        key: &JsonPath,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Vec<Value>> {
        let mut payload = self.get_payload_internal(point_id)?;
        
        // Remove using JsonPath
        let deleted_values = payload.remove(key);
        
        self.set_payload_internal(point_id, &payload)?;
        Ok(deleted_values)
    }
    
    fn clear(
        &mut self,
        point_id: PointOffsetType,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<Payload>> {
        let existing = self.get_payload_internal(point_id).ok();
        
        // Set empty payload
        let empty = Payload::default();
        self.set_payload_internal(point_id, &empty)?;
        
        // Clear cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.remove(&point_id);
        }
        
        Ok(existing)
    }
    
    #[cfg(test)]
    fn clear_all(&mut self, _hw_counter: &HardwareCounterCell) -> OperationResult<()> {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        Ok(())
    }
    
    fn flusher(&self) -> Flusher {
        // Since VDECollection pointer can't be sent between threads,
        // we return a no-op flusher. VDE flushes on Drop.
        Box::new(|| Ok(()))
    }
    
    fn iter<F>(&self, mut callback: F, _hw_counter: &HardwareCounterCell) -> OperationResult<()>
    where
        F: FnMut(PointOffsetType, &Payload) -> OperationResult<bool>,
    {
        // Iterate over cache (VDE doesn't expose iteration API yet)
        let cache = self.cache.read().unwrap();
        
        for (point_id, payload) in cache.iter() {
            let should_continue = callback(*point_id, payload)?;
            if !should_continue {
                break;
            }
        }
        
        Ok(())
    }
    
    fn files(&self) -> Vec<PathBuf> {
        vec![
            self.path.join(format!("{}_metadata.btr", self.name)),
        ]
    }
    
    fn get_storage_size_bytes(&self) -> OperationResult<usize> {
        // Estimate based on cache size
        let cache = self.cache.read().unwrap();
        let estimated_size: usize = cache.iter()
            .map(|(_, payload)| {
                serde_json::to_string(payload).unwrap_or_default().len()
            })
            .sum();
        
        Ok(estimated_size)
    }
    
    fn is_on_disk(&self) -> bool {
        true // VDE uses Btrieve2 for persistent storage
    }
}

impl Drop for VDEPayloadStorage {
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

unsafe impl Send for VDEPayloadStorage {}
unsafe impl Sync for VDEPayloadStorage {}
