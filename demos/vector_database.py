#!/usr/bin/env python3
"""
Production-Ready Vector Database Implementation with FAISS HNSW

A complete VectorDB implementation featuring:
- CRUD operations (Create, Read, Update, Delete)
- Multiple index types with automatic selection
- Metadata storage and filtering
- Batch operations
- Persistence (save/load)
- Automatic consolidation
- Thread-safe operations
"""

import numpy as np
import faiss
import pickle
import json
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time
from pathlib import Path


class IndexType(Enum):
    """Supported index types."""
    FLAT = "Flat"  # Exact search, supports deletion
    HNSW = "HNSW"  # Fast approximate search
    IVF_FLAT = "IVFFlat"  # Balanced, supports deletion


@dataclass
class VectorMetadata:
    """Metadata associated with each vector."""
    id: str  # External ID (e.g., document ID)
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class VectorDatabase:
    """
    Production-ready Vector Database with HNSW and deletion support.
    
    Features:
    - CRUD operations
    - Multiple index types
    - Metadata storage
    - Persistence
    - Automatic consolidation
    - Thread-safe
    
    Example:
        >>> db = VectorDatabase(dimension=384, index_type=IndexType.HNSW)
        >>> db.add(vectors, metadata=metadata_list)
        >>> results = db.search(query_vector, k=10)
        >>> db.delete(['id1', 'id2'])
        >>> db.save('vectordb.faiss')
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: IndexType = IndexType.HNSW,
        metric: str = 'L2',
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 40,
        hnsw_ef_search: int = 16,
        consolidate_threshold: float = 0.25,
        auto_consolidate: bool = True
    ):
        """
        Initialize Vector Database.
        
        Args:
            dimension: Vector dimension
            index_type: Type of FAISS index to use
            metric: Distance metric ('L2' or 'IP')
            hnsw_m: HNSW M parameter
            hnsw_ef_construction: HNSW construction effort
            hnsw_ef_search: HNSW search effort
            consolidate_threshold: Auto-consolidate when deletion ratio exceeds this
            auto_consolidate: Whether to automatically consolidate
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.consolidate_threshold = consolidate_threshold
        self.auto_consolidate = auto_consolidate
        
        # Create FAISS index
        metric_type = faiss.METRIC_L2 if metric == 'L2' else faiss.METRIC_INNER_PRODUCT
        
        if index_type == IndexType.HNSW:
            self.index = faiss.IndexHNSWFlat(dimension, hnsw_m, metric_type)
            self.index.hnsw.efConstruction = hnsw_ef_construction
            self.index.hnsw.efSearch = hnsw_ef_search
        elif index_type == IndexType.FLAT:
            self.index = faiss.IndexFlatL2(dimension) if metric == 'L2' else faiss.IndexFlatIP(dimension)
        elif index_type == IndexType.IVF_FLAT:
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
            self.index.nprobe = 10
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # ID management
        self.external_to_internal: Dict[str, int] = {}  # external_id -> internal_id
        self.internal_to_external: Dict[int, str] = {}  # internal_id -> external_id
        self.next_internal_id = 0
        
        # Metadata storage
        self.metadata: Dict[str, VectorMetadata] = {}
        
        # Deletion tracking
        self.deleted_external_ids: set = set()
        
        # Vector storage for consolidation
        self.vectors: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.stats = {
            'total_added': 0,
            'total_deleted': 0,
            'total_searches': 0,
            'last_consolidation': None,
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Training status (for IVF indexes)
        self.is_trained = index_type != IndexType.IVF_FLAT
    
    def add(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[VectorMetadata]] = None,
        batch_size: int = 1000
    ) -> List[str]:
        """
        Add vectors to the database.
        
        Args:
            vectors: Array of vectors (n, d) or list of vectors
            ids: Optional external IDs. If None, auto-generated
            metadata: Optional metadata for each vector
            batch_size: Batch size for adding vectors
            
        Returns:
            List of IDs assigned to vectors
        """
        with self.lock:
            # Convert to numpy array
            if isinstance(vectors, list):
                vectors = np.array(vectors, dtype=np.float32)
            elif vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            n = vectors.shape[0]
            assert vectors.shape[1] == self.dimension, \
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}"
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{self.stats['total_added'] + i}" for i in range(n)]
            else:
                assert len(ids) == n, "Number of IDs must match number of vectors"
            
            # Check for duplicate IDs
            for id_ in ids:
                if id_ in self.external_to_internal and id_ not in self.deleted_external_ids:
                    raise ValueError(f"ID {id_} already exists. Use update() to modify.")
            
            # Train index if needed (IVF)
            if not self.is_trained:
                print(f"Training index with {n} vectors...")
                self.index.train(vectors)
                self.is_trained = True
            
            # Add vectors in batches
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch_vectors = vectors[i:end]
                batch_ids = ids[i:end]
                batch_metadata = metadata[i:end] if metadata else None
                
                # Add to FAISS index
                self.index.add(batch_vectors)
                
                # Update mappings
                for j, external_id in enumerate(batch_ids):
                    internal_id = self.next_internal_id + j
                    self.external_to_internal[external_id] = internal_id
                    self.internal_to_external[internal_id] = external_id
                    self.vectors[external_id] = batch_vectors[j].copy()
                    
                    # Store metadata
                    if batch_metadata and batch_metadata[j]:
                        self.metadata[external_id] = batch_metadata[j]
                    else:
                        self.metadata[external_id] = VectorMetadata(
                            id=external_id,
                            timestamp=time.time()
                        )
                
                self.next_internal_id += end - i
            
            self.stats['total_added'] += n
            
            return ids
    
    def search(
        self,
        query: Union[np.ndarray, List[float]],
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[float], List[VectorMetadata]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filter (key-value pairs)
            
        Returns:
            ids: List of matching IDs
            distances: List of distances
            metadata: List of metadata objects
        """
        with self.lock:
            # Convert query to numpy array
            if isinstance(query, list):
                query = np.array([query], dtype=np.float32)
            elif query.ndim == 1:
                query = query.reshape(1, -1).astype(np.float32)
            
            # Calculate how many results to fetch
            deleted_ratio = len(self.deleted_external_ids) / max(self.index.ntotal, 1)
            k_search = k
            
            if deleted_ratio > 0.01 or filter_metadata:
                # Fetch more to account for deletions and filtering
                k_search = min(int(k * (1 + deleted_ratio * 3) + 20), self.index.ntotal)
            
            # Perform search
            distances, internal_ids = self.index.search(query, k_search)
            
            # Convert to external IDs and filter
            result_ids = []
            result_distances = []
            result_metadata = []
            
            for j in range(k_search):
                internal_id = internal_ids[0, j]
                
                if internal_id < 0 or internal_id >= self.next_internal_id:
                    continue
                
                external_id = self.internal_to_external.get(internal_id)
                
                if external_id is None or external_id in self.deleted_external_ids:
                    continue
                
                # Apply metadata filter
                if filter_metadata:
                    meta = self.metadata.get(external_id)
                    if not meta or not self._matches_filter(meta, filter_metadata):
                        continue
                
                result_ids.append(external_id)
                result_distances.append(float(distances[0, j]))
                result_metadata.append(self.metadata.get(external_id))
                
                if len(result_ids) >= k:
                    break
            
            self.stats['total_searches'] += 1
            
            return result_ids, result_distances, result_metadata
    
    def delete(self, ids: Union[str, List[str]]) -> int:
        """
        Delete vectors by ID.
        
        Args:
            ids: Single ID or list of IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        with self.lock:
            if isinstance(ids, str):
                ids = [ids]
            
            deleted_count = 0
            for external_id in ids:
                if (external_id in self.external_to_internal and 
                    external_id not in self.deleted_external_ids):
                    self.deleted_external_ids.add(external_id)
                    deleted_count += 1
            
            self.stats['total_deleted'] += deleted_count
            
            # Check if consolidation is needed
            if self.auto_consolidate:
                deleted_ratio = len(self.deleted_external_ids) / max(self.index.ntotal, 1)
                if deleted_ratio > self.consolidate_threshold:
                    print(f"Auto-consolidating (deletion ratio: {deleted_ratio:.1%})")
                    self.consolidate()
            
            return deleted_count
    
    def update(
        self,
        id: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[VectorMetadata] = None,
        strategy: str = 'lazy'
    ) -> bool:
        """
        Update a vector and/or its metadata.
        
        HNSW不支持原地更新，提供三种策略：
        
        1. 'lazy' (推荐): 懒更新 - 标记旧向量为删除，添加新向量
           - 优点：快速，O(log n)
           - 缺点：产生碎片，需要定期consolidate
           - 适用：频繁更新，可容忍碎片
        
        2. 'reconstruct': 重建策略 - 删除旧向量，添加新向量，保持ID不变
           - 优点：内部ID保持一致，易于追踪
           - 缺点：需要维护ID映射
           - 适用：中等更新频率
        
        3. 'batch': 批量更新 - 累积更新，批量重建
           - 优点：适合大量更新
           - 缺点：更新不实时
           - 适用：批处理场景
        
        Args:
            id: External ID of vector to update
            vector: New vector (if updating vector)
            metadata: New metadata (if updating metadata)
            strategy: Update strategy ('lazy', 'reconstruct', 'batch')
            
        Returns:
            True if updated, False if ID not found
        """
        with self.lock:
            if id not in self.external_to_internal:
                return False
            
            if id in self.deleted_external_ids:
                return False
            
            # Metadata-only update (always fast)
            if metadata is not None:
                self.metadata[id] = metadata
                if vector is None:
                    return True  # Only metadata update, done
            
            # Vector update strategies
            if vector is not None:
                if vector.ndim == 1:
                    vector = vector.reshape(1, -1)
                vector = vector.astype(np.float32)
                
                if strategy == 'lazy':
                    return self._update_lazy(id, vector)
                elif strategy == 'reconstruct':
                    return self._update_reconstruct(id, vector)
                elif strategy == 'batch':
                    return self._update_batch(id, vector)
                else:
                    raise ValueError(f"Unknown update strategy: {strategy}")
            
            return True
    
    def _update_lazy(self, id: str, vector: np.ndarray) -> bool:
        """
        懒更新策略：标记旧的删除，添加新的
        
        工作原理：
        1. 把旧的internal_id标记为删除
        2. 添加新向量，获得新的internal_id
        3. 更新映射关系
        
        优点：快速，不需要重建索引
        缺点：产生"墓碑"，占用内存
        """
        # Get old internal ID
        old_internal_id = self.external_to_internal[id]
        
        # Mark old vector as deleted (by adding to deleted set temporarily)
        # We use a special approach: add new vector first, then mark old as deleted
        
        # Add new vector to index
        self.index.add(vector)
        new_internal_id = self.next_internal_id
        self.next_internal_id += 1
        
        # Update mappings to point to new internal ID
        self.external_to_internal[id] = new_internal_id
        self.internal_to_external[new_internal_id] = id
        
        # Clean up old internal ID from mapping
        if old_internal_id in self.internal_to_external:
            del self.internal_to_external[old_internal_id]
        
        # Update stored vector
        self.vectors[id] = vector[0].copy()
        
        # Note: old internal_id becomes orphaned (not in internal_to_external)
        # It will be filtered out during search
        
        return True
    
    def _update_reconstruct(self, id: str, vector: np.ndarray) -> bool:
        """
        重建策略：删除后重新添加，保持external ID
        
        工作原理：
        1. 标记external_id为删除
        2. 添加新向量
        3. 从删除集合移除
        
        优点：逻辑清晰，利用现有删除机制
        缺点：internal_id会改变
        """
        # Mark as deleted
        self.deleted_external_ids.add(id)
        
        # Add new vector
        self.index.add(vector)
        new_internal_id = self.next_internal_id
        self.next_internal_id += 1
        
        # Update mappings
        old_internal_id = self.external_to_internal[id]
        self.external_to_internal[id] = new_internal_id
        
        # Clean up old mapping
        if old_internal_id in self.internal_to_external:
            del self.internal_to_external[old_internal_id]
        
        # Add new mapping
        self.internal_to_external[new_internal_id] = id
        
        # Remove from deleted set (update complete)
        self.deleted_external_ids.discard(id)
        
        # Update stored vector
        self.vectors[id] = vector[0].copy()
        
        return True
    
    def _update_batch(self, id: str, vector: np.ndarray) -> bool:
        """
        批量更新策略：先标记，稍后批量处理
        
        这个方法需要配合 apply_batch_updates() 使用
        """
        # For now, fall back to lazy update
        # In production, you might want to maintain a pending_updates queue
        return self._update_lazy(id, vector)
    
    def batch_update(
        self,
        updates: List[Tuple[str, np.ndarray]],
        metadata_updates: Optional[Dict[str, VectorMetadata]] = None
    ) -> int:
        """
        批量更新多个向量（高效）
        
        Args:
            updates: List of (id, vector) tuples
            metadata_updates: Optional dict of id -> metadata
            
        Returns:
            Number of vectors successfully updated
        """
        with self.lock:
            updated_count = 0
            
            # Collect all vectors to add
            vectors_to_add = []
            ids_to_update = []
            old_internal_ids = []
            
            for ext_id, vector in updates:
                if ext_id not in self.external_to_internal:
                    continue
                if ext_id in self.deleted_external_ids:
                    continue
                
                # Prepare vector
                if vector.ndim == 1:
                    vector = vector.reshape(1, -1)
                vector = vector.astype(np.float32)
                
                vectors_to_add.append(vector[0])
                ids_to_update.append(ext_id)
                old_internal_ids.append(self.external_to_internal[ext_id])
            
            if not vectors_to_add:
                return 0
            
            # Batch add all new vectors
            vectors_array = np.array(vectors_to_add, dtype=np.float32)
            self.index.add(vectors_array)
            
            # Update all mappings
            start_internal_id = self.next_internal_id
            for i, ext_id in enumerate(ids_to_update):
                new_internal_id = start_internal_id + i
                
                # Update mappings
                old_internal_id = old_internal_ids[i]
                self.external_to_internal[ext_id] = new_internal_id
                self.internal_to_external[new_internal_id] = ext_id
                
                # Clean up old mapping
                if old_internal_id in self.internal_to_external:
                    del self.internal_to_external[old_internal_id]
                
                # Update stored vector
                self.vectors[ext_id] = vectors_array[i].copy()
                
                # Update metadata if provided
                if metadata_updates and ext_id in metadata_updates:
                    self.metadata[ext_id] = metadata_updates[ext_id]
                
                updated_count += 1
            
            self.next_internal_id += len(ids_to_update)
            
            return updated_count
    
    def get(self, id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """
        Get vector and metadata by ID.
        
        Args:
            id: External ID
            
        Returns:
            (vector, metadata) tuple or None if not found
        """
        with self.lock:
            if id in self.deleted_external_ids:
                return None
            
            if id not in self.external_to_internal:
                return None
            
            vector = self.vectors.get(id)
            metadata = self.metadata.get(id)
            
            return (vector, metadata)
    
    def consolidate(self, verbose: bool = True) -> None:
        """
        Rebuild index without deleted vectors.
        
        This is an expensive operation but reclaims memory and improves performance.
        Should be called during low-traffic periods.
        """
        with self.lock:
            if len(self.deleted_external_ids) == 0:
                if verbose:
                    print("No deleted vectors to consolidate")
                return
            
            start_time = time.time()
            
            if verbose:
                print(f"Consolidating: removing {len(self.deleted_external_ids)} "
                      f"deleted vectors from {self.index.ntotal} total...")
            
            # Collect valid vectors and IDs
            valid_vectors = []
            valid_external_ids = []
            
            for ext_id in self.external_to_internal.keys():
                if ext_id not in self.deleted_external_ids:
                    vector = self.vectors.get(ext_id)
                    if vector is not None:
                        valid_vectors.append(vector)
                        valid_external_ids.append(ext_id)
            
            if not valid_vectors:
                return
            
            valid_vectors = np.array(valid_vectors, dtype=np.float32)
            
            # Create new index with same parameters
            metric_type = faiss.METRIC_L2 if self.metric == 'L2' else faiss.METRIC_INNER_PRODUCT
            
            if self.index_type == IndexType.HNSW:
                new_index = faiss.IndexHNSWFlat(
                    self.dimension,
                    self.index.hnsw.cum_nneighbor_per_level[1] // 2,
                    metric_type
                )
                new_index.hnsw.efConstruction = self.index.hnsw.efConstruction
                new_index.hnsw.efSearch = self.index.hnsw.efSearch
            elif self.index_type == IndexType.FLAT:
                new_index = faiss.IndexFlatL2(self.dimension) if self.metric == 'L2' \
                    else faiss.IndexFlatIP(self.dimension)
            elif self.index_type == IndexType.IVF_FLAT:
                nlist = self.index.nlist
                quantizer = faiss.IndexFlatL2(self.dimension)
                new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric_type)
                new_index.train(valid_vectors)
                new_index.nprobe = self.index.nprobe
            
            # Add valid vectors
            new_index.add(valid_vectors)
            
            # Rebuild mappings
            new_external_to_internal = {}
            new_internal_to_external = {}
            
            for i, ext_id in enumerate(valid_external_ids):
                new_external_to_internal[ext_id] = i
                new_internal_to_external[i] = ext_id
            
            # Update database
            self.index = new_index
            self.external_to_internal = new_external_to_internal
            self.internal_to_external = new_internal_to_external
            self.next_internal_id = len(valid_external_ids)
            
            # Clean up deleted vectors and metadata
            for ext_id in self.deleted_external_ids:
                self.vectors.pop(ext_id, None)
                self.metadata.pop(ext_id, None)
            
            self.deleted_external_ids.clear()
            
            elapsed = time.time() - start_time
            self.stats['last_consolidation'] = time.time()
            
            if verbose:
                print(f"Consolidation complete in {elapsed:.2f}s. "
                      f"Index now has {self.index.ntotal} vectors")
    
    def save(self, path: str) -> None:
        """
        Save database to disk.
        
        Args:
            path: Directory path to save database
        """
        with self.lock:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(path / "index.faiss"))
            
            # Save metadata and mappings
            db_state = {
                'dimension': self.dimension,
                'index_type': self.index_type.value,
                'metric': self.metric,
                'external_to_internal': self.external_to_internal,
                'internal_to_external': {int(k): v for k, v in self.internal_to_external.items()},
                'next_internal_id': self.next_internal_id,
                'deleted_external_ids': list(self.deleted_external_ids),
                'stats': self.stats,
                'consolidate_threshold': self.consolidate_threshold,
            }
            
            with open(path / "db_state.json", 'w') as f:
                json.dump(db_state, f, indent=2)
            
            # Save metadata
            metadata_dict = {
                id_: asdict(meta) for id_, meta in self.metadata.items()
            }
            with open(path / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Save vectors
            with open(path / "vectors.pkl", 'wb') as f:
                pickle.dump(self.vectors, f)
            
            print(f"Database saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'VectorDatabase':
        """
        Load database from disk.
        
        Args:
            path: Directory path containing saved database
            
        Returns:
            Loaded VectorDatabase instance
        """
        path = Path(path)
        
        # Load state
        with open(path / "db_state.json", 'r') as f:
            db_state = json.load(f)
        
        # Create instance
        db = cls(
            dimension=db_state['dimension'],
            index_type=IndexType(db_state['index_type']),
            metric=db_state['metric'],
            consolidate_threshold=db_state.get('consolidate_threshold', 0.25)
        )
        
        # Load FAISS index
        db.index = faiss.read_index(str(path / "index.faiss"))
        db.is_trained = True
        
        # Restore state
        db.external_to_internal = db_state['external_to_internal']
        db.internal_to_external = {int(k): v for k, v in db_state['internal_to_external'].items()}
        db.next_internal_id = db_state['next_internal_id']
        db.deleted_external_ids = set(db_state['deleted_external_ids'])
        db.stats = db_state['stats']
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata_dict = json.load(f)
        db.metadata = {
            id_: VectorMetadata(**meta) for id_, meta in metadata_dict.items()
        }
        
        # Load vectors
        with open(path / "vectors.pkl", 'rb') as f:
            db.vectors = pickle.load(f)
        
        print(f"Database loaded from {path}")
        print(f"  Vectors: {db.index.ntotal}")
        print(f"  Deleted: {len(db.deleted_external_ids)}")
        
        return db
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.lock:
            return {
                'total_vectors': self.index.ntotal,
                'deleted_vectors': len(self.deleted_external_ids),
                'active_vectors': self.index.ntotal - len(self.deleted_external_ids),
                'deletion_ratio': len(self.deleted_external_ids) / max(self.index.ntotal, 1),
                **self.stats
            }
    
    def _matches_filter(self, metadata: VectorMetadata, filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        if metadata.metadata is None:
            return False
        
        for key, value in filter_dict.items():
            if key not in metadata.metadata or metadata.metadata[key] != value:
                return False
        
        return True


def demo():
    """Demonstration of VectorDatabase usage."""
    print("=" * 70)
    print("Production Vector Database Demo")
    print("=" * 70)
    
    # Create database
    db = VectorDatabase(
        dimension=128,
        index_type=IndexType.HNSW,
        hnsw_m=32,
        consolidate_threshold=0.25
    )
    
    print("\n1. Adding vectors with metadata...")
    vectors = np.random.random((1000, 128)).astype('float32')
    metadata_list = [
        VectorMetadata(
            id=f"doc_{i}",
            text=f"Document {i}",
            metadata={'category': 'A' if i % 2 == 0 else 'B', 'score': i}
        )
        for i in range(1000)
    ]
    ids = db.add(vectors, metadata=metadata_list)
    print(f"   Added {len(ids)} vectors")
    print(f"   Stats: {db.get_stats()}")
    
    print("\n2. Searching...")
    query = np.random.random(128).astype('float32')
    result_ids, distances, result_metadata = db.search(query, k=5)
    print(f"   Top 5 results:")
    for id_, dist, meta in zip(result_ids, distances, result_metadata):
        print(f"     {id_}: distance={dist:.4f}, category={meta.metadata['category']}")
    
    print("\n3. Searching with metadata filter...")
    result_ids, distances, result_metadata = db.search(
        query, k=5, filter_metadata={'category': 'A'}
    )
    print(f"   Top 5 results (category A only):")
    for id_, dist, meta in zip(result_ids, distances, result_metadata):
        print(f"     {id_}: distance={dist:.4f}")
    
    print("\n4. Deleting vectors...")
    ids_to_delete = [f"doc_{i}" for i in range(0, 100)]
    deleted = db.delete(ids_to_delete)
    print(f"   Deleted {deleted} vectors")
    print(f"   Stats: {db.get_stats()}")
    
    print("\n5. Updating a vector (lazy strategy)...")
    new_vector = np.random.random(128).astype('float32')
    db.update('doc_100', vector=new_vector, strategy='lazy')
    print("   Updated doc_100 (lazy)")
    print(f"   Stats after update: {db.get_stats()}")
    
    print("\n5b. Batch updating multiple vectors...")
    updates = [(f"doc_{i}", np.random.random(128).astype('float32')) for i in range(200, 210)]
    updated = db.batch_update(updates)
    print(f"   Batch updated {updated} vectors")
    print(f"   Stats after batch update: {db.get_stats()}")
    
    print("\n6. Getting a specific vector...")
    vector, metadata = db.get('doc_100')
    print(f"   Retrieved: {metadata.id}, shape: {vector.shape}")
    
    print("\n7. Saving database...")
    db.save('/tmp/my_vectordb')
    
    print("\n8. Loading database...")
    db2 = VectorDatabase.load('/tmp/my_vectordb')
    print(f"   Loaded stats: {db2.get_stats()}")
    
    print("\n9. Testing loaded database with search...")
    result_ids, distances, _ = db2.search(query, k=5)
    print(f"   Search results: {result_ids[:3]}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == '__main__':
    demo()
