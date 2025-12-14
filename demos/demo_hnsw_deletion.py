#!/usr/bin/env python3
"""
Demo: HNSW Index with Deletion Support for Vector Database

This demo shows multiple strategies to implement deletion in FAISS HNSW index:
1. Lazy deletion with tombstone (recommended)
2. Rebuild strategy
3. Hybrid approach with periodic consolidation

Perfect for building a production VectorDB with HNSW!
"""

import numpy as np
import faiss
import time
from typing import List, Set, Optional, Tuple


class HNSWWithDeletion:
    """
    HNSW Index wrapper with lazy deletion support using tombstone marking.
    
    This is the recommended approach for most production use cases.
    Deleted IDs are marked but not removed until consolidate() is called.
    """
    
    def __init__(self, d: int, M: int = 32, metric: str = 'L2', 
                 consolidate_threshold: float = 0.3):
        """
        Args:
            d: Vector dimension
            M: HNSW M parameter (number of neighbors)
            metric: 'L2' or 'IP' (inner product)
            consolidate_threshold: Trigger consolidation when deletion ratio exceeds this
        """
        self.d = d
        self.M = M
        self.metric = faiss.METRIC_L2 if metric == 'L2' else faiss.METRIC_INNER_PRODUCT
        self.consolidate_threshold = consolidate_threshold
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(d, M, self.metric)
        
        # Tombstone tracking
        self.deleted_ids: Set[int] = set()
        self.total_added = 0
        
        # Store vectors for consolidation (optional, uses more memory)
        self.store_vectors = True
        self.vectors = []  # List of (id, vector) tuples
        
    def add(self, vectors: np.ndarray) -> List[int]:
        """
        Add vectors to index.
        
        Args:
            vectors: numpy array of shape (n, d)
            
        Returns:
            List of assigned IDs
        """
        n = vectors.shape[0]
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        # Store vectors if needed
        if self.store_vectors:
            for i in range(n):
                self.vectors.append((start_id + i, vectors[i].copy()))
        
        self.total_added += n
        assigned_ids = list(range(start_id, start_id + n))
        return assigned_ids
    
    def remove(self, ids_to_remove: List[int]) -> int:
        """
        Mark vectors as deleted (lazy deletion).
        
        Args:
            ids_to_remove: List of IDs to delete
            
        Returns:
            Number of IDs actually deleted (excluding already deleted)
        """
        new_deletions = 0
        for id_val in ids_to_remove:
            if id_val not in self.deleted_ids and id_val < self.index.ntotal:
                self.deleted_ids.add(id_val)
                new_deletions += 1
        
        # Check if consolidation is recommended
        deleted_ratio = len(self.deleted_ids) / max(self.index.ntotal, 1)
        if deleted_ratio > self.consolidate_threshold:
            print(f"Warning: Deletion ratio {deleted_ratio:.1%} exceeds threshold. "
                  f"Consider calling consolidate()")
        
        return new_deletions
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors, filtering out deleted vectors.
        
        Args:
            queries: numpy array of shape (n, d)
            k: number of neighbors to return
            
        Returns:
            distances: array of shape (n, k)
            ids: array of shape (n, k)
        """
        n = queries.shape[0]
        deleted_ratio = len(self.deleted_ids) / max(self.index.ntotal, 1)
        
        # Fetch more results to account for deleted ones
        k_search = k
        if deleted_ratio > 0.01:
            # Dynamic k_search based on deletion ratio
            k_search = min(int(k * (1 + deleted_ratio * 3)), self.index.ntotal)
            k_search = max(k_search, k + 10)
        
        # Search the index
        distances, ids = self.index.search(queries, k_search)
        
        # Filter out deleted IDs
        final_distances = np.full((n, k), np.inf, dtype=np.float32)
        final_ids = np.full((n, k), -1, dtype=np.int64)
        
        for i in range(n):
            valid_count = 0
            for j in range(k_search):
                id_val = ids[i, j]
                if id_val >= 0 and id_val not in self.deleted_ids:
                    final_distances[i, valid_count] = distances[i, j]
                    final_ids[i, valid_count] = id_val
                    valid_count += 1
                    if valid_count >= k:
                        break
        
        return final_distances, final_ids
    
    def consolidate(self, verbose: bool = True) -> None:
        """
        Rebuild the index without deleted vectors.
        This is an expensive operation but reduces memory and improves performance.
        """
        if len(self.deleted_ids) == 0:
            if verbose:
                print("No deleted vectors to consolidate.")
            return
        
        if verbose:
            print(f"Consolidating: removing {len(self.deleted_ids)} deleted vectors "
                  f"from {self.index.ntotal} total...")
        
        start_time = time.time()
        
        # Collect valid vectors
        if self.store_vectors:
            # Use stored vectors (faster)
            valid_vectors = []
            id_mapping = {}  # old_id -> new_id
            new_id = 0
            
            for old_id, vector in self.vectors:
                if old_id not in self.deleted_ids:
                    valid_vectors.append(vector)
                    id_mapping[old_id] = new_id
                    new_id += 1
            
            valid_vectors = np.array(valid_vectors, dtype=np.float32)
        else:
            # Reconstruct from index (slower)
            valid_vectors = []
            id_mapping = {}
            new_id = 0
            
            for old_id in range(self.index.ntotal):
                if old_id not in self.deleted_ids:
                    vector = self.index.reconstruct(old_id)
                    valid_vectors.append(vector)
                    id_mapping[old_id] = new_id
                    new_id += 1
            
            valid_vectors = np.array(valid_vectors, dtype=np.float32)
        
        # Create new index with same parameters
        new_index = faiss.IndexHNSWFlat(self.d, self.M, self.metric)
        new_index.hnsw.efConstruction = self.index.hnsw.efConstruction
        new_index.hnsw.efSearch = self.index.hnsw.efSearch
        
        # Add valid vectors
        if len(valid_vectors) > 0:
            new_index.add(valid_vectors)
        
        # Update vectors storage
        if self.store_vectors:
            self.vectors = [(i, valid_vectors[i]) for i in range(len(valid_vectors))]
        
        # Replace index
        self.index = new_index
        self.deleted_ids.clear()
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Consolidation complete in {elapsed:.2f}s. "
                  f"Index now has {self.index.ntotal} vectors.")
    
    def get_stats(self) -> dict:
        """Get statistics about the index."""
        return {
            'total_vectors': self.index.ntotal,
            'deleted_vectors': len(self.deleted_ids),
            'active_vectors': self.index.ntotal - len(self.deleted_ids),
            'deletion_ratio': len(self.deleted_ids) / max(self.index.ntotal, 1),
            'total_added': self.total_added,
        }
    
    def set_ef_search(self, ef: int):
        """Set the search effort parameter."""
        self.index.hnsw.efSearch = ef
    
    def set_ef_construction(self, ef: int):
        """Set the construction effort parameter."""
        self.index.hnsw.efConstruction = ef


class HNSWIDMapWithDeletion:
    """
    HNSW with custom ID mapping for easier deletion tracking.
    
    Use this when you need to maintain external IDs (e.g., database primary keys).
    """
    
    def __init__(self, d: int, M: int = 32, metric: str = 'L2'):
        self.d = d
        self.M = M
        self.metric = faiss.METRIC_L2 if metric == 'L2' else faiss.METRIC_INNER_PRODUCT
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(d, M, self.metric)
        
        # ID mapping
        self.id_map = []  # internal_id -> external_id
        self.reverse_map = {}  # external_id -> internal_id
        self.deleted_external_ids: Set[int] = set()
        
        # Vector storage for consolidation
        self.vectors = {}  # external_id -> vector
    
    def add_with_ids(self, vectors: np.ndarray, external_ids: List[int]) -> int:
        """
        Add vectors with custom external IDs.
        
        Args:
            vectors: numpy array of shape (n, d)
            external_ids: List of custom IDs
            
        Returns:
            Number of vectors added
        """
        n = vectors.shape[0]
        assert len(external_ids) == n, "Number of IDs must match number of vectors"
        
        start_internal_id = self.index.ntotal
        self.index.add(vectors)
        
        # Update mappings
        for i, ext_id in enumerate(external_ids):
            internal_id = start_internal_id + i
            self.id_map.append(ext_id)
            self.reverse_map[ext_id] = internal_id
            self.vectors[ext_id] = vectors[i].copy()
        
        return n
    
    def remove(self, external_ids: List[int]) -> int:
        """
        Delete vectors by external IDs.
        
        Args:
            external_ids: List of external IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        deleted_count = 0
        for ext_id in external_ids:
            if ext_id in self.reverse_map and ext_id not in self.deleted_external_ids:
                self.deleted_external_ids.add(ext_id)
                deleted_count += 1
        
        return deleted_count
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search and return results with external IDs.
        
        Args:
            queries: numpy array of shape (n, d)
            k: number of neighbors
            
        Returns:
            distances: array of shape (n, k)
            external_ids: array of shape (n, k)
        """
        n = queries.shape[0]
        deleted_ratio = len(self.deleted_external_ids) / max(self.index.ntotal, 1)
        
        # Fetch extra results
        k_search = min(int(k * (1 + deleted_ratio * 3) + 10), self.index.ntotal)
        
        distances, internal_ids = self.index.search(queries, k_search)
        
        # Map to external IDs and filter deleted
        final_distances = np.full((n, k), np.inf, dtype=np.float32)
        final_ids = np.full((n, k), -1, dtype=np.int64)
        
        for i in range(n):
            valid_count = 0
            for j in range(k_search):
                internal_id = internal_ids[i, j]
                if 0 <= internal_id < len(self.id_map):
                    external_id = self.id_map[internal_id]
                    if external_id not in self.deleted_external_ids:
                        final_distances[i, valid_count] = distances[i, j]
                        final_ids[i, valid_count] = external_id
                        valid_count += 1
                        if valid_count >= k:
                            break
        
        return final_distances, final_ids
    
    def consolidate(self, verbose: bool = True) -> None:
        """Rebuild index without deleted vectors."""
        if len(self.deleted_external_ids) == 0:
            return
        
        if verbose:
            print(f"Consolidating: removing {len(self.deleted_external_ids)} deleted vectors...")
        
        # Collect valid vectors and IDs
        valid_vectors = []
        valid_external_ids = []
        
        for ext_id, vector in self.vectors.items():
            if ext_id not in self.deleted_external_ids:
                valid_vectors.append(vector)
                valid_external_ids.append(ext_id)
        
        valid_vectors = np.array(valid_vectors, dtype=np.float32)
        
        # Create new index
        new_index = faiss.IndexHNSWFlat(self.d, self.M, self.metric)
        new_index.hnsw.efConstruction = self.index.hnsw.efConstruction
        new_index.hnsw.efSearch = self.index.hnsw.efSearch
        
        # Reset mappings
        self.id_map = []
        self.reverse_map = {}
        
        # Add valid vectors
        if len(valid_vectors) > 0:
            new_index.add(valid_vectors)
            for i, ext_id in enumerate(valid_external_ids):
                self.id_map.append(ext_id)
                self.reverse_map[ext_id] = i
        
        # Clean up deleted from vectors storage
        for ext_id in self.deleted_external_ids:
            if ext_id in self.vectors:
                del self.vectors[ext_id]
        
        self.index = new_index
        self.deleted_external_ids.clear()
        
        if verbose:
            print(f"Consolidation complete. Index now has {self.index.ntotal} vectors.")


def demo_basic_deletion():
    """Demo 1: Basic deletion with tombstone marking."""
    print("=" * 70)
    print("Demo 1: Basic Deletion with Tombstone Marking")
    print("=" * 70)
    
    d = 128
    nb = 10000
    nq = 10
    
    # Generate random data
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    # Create index with deletion support
    index = HNSWWithDeletion(d, M=32)
    index.set_ef_construction(40)
    index.set_ef_search(16)
    
    print(f"\n1. Adding {nb} vectors...")
    ids = index.add(xb)
    print(f"   Added vectors with IDs: 0 to {len(ids)-1}")
    print(f"   Stats: {index.get_stats()}")
    
    # Initial search
    print(f"\n2. Searching for {nq} queries (k=5)...")
    k = 5
    distances, labels = index.search(xq, k)
    print(f"   First query results: {labels[0]}")
    
    # Delete some vectors
    print(f"\n3. Deleting 1000 vectors (IDs 0-999)...")
    ids_to_delete = list(range(1000))
    deleted = index.remove(ids_to_delete)
    print(f"   Deleted: {deleted} vectors")
    print(f"   Stats: {index.get_stats()}")
    
    # Search after deletion
    print(f"\n4. Searching after deletion...")
    distances, labels = index.search(xq, k)
    print(f"   First query results: {labels[0]}")
    
    # Verify no deleted IDs in results
    deleted_in_results = any(id_val in ids_to_delete for id_val in labels.flatten() if id_val >= 0)
    print(f"   Deleted IDs in results: {'YES (BUG!)' if deleted_in_results else 'NO (Correct!)'}")
    
    # Delete more to trigger consolidation
    print(f"\n5. Deleting 2000 more vectors...")
    ids_to_delete2 = list(range(1000, 3000))
    index.remove(ids_to_delete2)
    print(f"   Stats: {index.get_stats()}")
    
    # Consolidate
    print(f"\n6. Consolidating index...")
    index.consolidate()
    print(f"   Stats after consolidation: {index.get_stats()}")
    
    print()


def demo_custom_ids():
    """Demo 2: Using custom external IDs."""
    print("=" * 70)
    print("Demo 2: Using Custom External IDs")
    print("=" * 70)
    
    d = 128
    nb = 1000
    nq = 5
    
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    # Create index with ID mapping
    index = HNSWIDMapWithDeletion(d, M=32)
    
    print(f"\n1. Adding {nb} vectors with custom IDs...")
    # Use custom IDs (e.g., database primary keys)
    custom_ids = [1000000 + i for i in range(nb)]
    index.add_with_ids(xb, custom_ids)
    print(f"   Added vectors with custom IDs: {custom_ids[0]} to {custom_ids[-1]}")
    
    print(f"\n2. Searching (k=5)...")
    k = 5
    distances, labels = index.search(xq, k)
    print(f"   First query results (custom IDs): {labels[0]}")
    
    print(f"\n3. Deleting vectors with IDs 1000000-1000099...")
    ids_to_delete = list(range(1000000, 1000100))
    deleted = index.remove(ids_to_delete)
    print(f"   Deleted: {deleted} vectors")
    
    print(f"\n4. Searching after deletion...")
    distances, labels = index.search(xq, k)
    print(f"   First query results: {labels[0]}")
    
    deleted_in_results = any(id_val in ids_to_delete for id_val in labels.flatten() if id_val >= 0)
    print(f"   Deleted IDs in results: {'YES (BUG!)' if deleted_in_results else 'NO (Correct!)'}")
    
    print()


def demo_performance():
    """Demo 3: Performance comparison."""
    print("=" * 70)
    print("Demo 3: Performance Characteristics")
    print("=" * 70)
    
    d = 128
    nb = 50000
    nq = 1000
    k = 10
    
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    index = HNSWWithDeletion(d, M=32, consolidate_threshold=0.3)
    index.set_ef_construction(40)
    index.set_ef_search(32)
    
    print(f"\n1. Building index with {nb} vectors...")
    start = time.time()
    index.add(xb)
    build_time = time.time() - start
    print(f"   Build time: {build_time:.2f}s ({nb/build_time:.0f} vectors/s)")
    
    print(f"\n2. Search performance (before deletion)...")
    start = time.time()
    distances, labels = index.search(xq, k)
    search_time = time.time() - start
    print(f"   Search time: {search_time*1000:.2f}ms ({nq/search_time:.0f} queries/s)")
    
    print(f"\n3. Deleting 10% of vectors...")
    ids_to_delete = list(range(0, nb, 10))  # Every 10th vector
    start = time.time()
    index.remove(ids_to_delete)
    delete_time = time.time() - start
    print(f"   Delete time: {delete_time*1000:.2f}ms (instant - lazy deletion)")
    print(f"   Stats: {index.get_stats()}")
    
    print(f"\n4. Search performance (after 10% deletion)...")
    start = time.time()
    distances, labels = index.search(xq, k)
    search_time_after = time.time() - start
    print(f"   Search time: {search_time_after*1000:.2f}ms ({nq/search_time_after:.0f} queries/s)")
    print(f"   Slowdown: {(search_time_after/search_time - 1)*100:.1f}%")
    
    print(f"\n5. Consolidation performance...")
    start = time.time()
    index.consolidate(verbose=False)
    consolidate_time = time.time() - start
    print(f"   Consolidation time: {consolidate_time:.2f}s")
    print(f"   Stats: {index.get_stats()}")
    
    print(f"\n6. Search performance (after consolidation)...")
    start = time.time()
    distances, labels = index.search(xq, k)
    search_time_consolidated = time.time() - start
    print(f"   Search time: {search_time_consolidated*1000:.2f}ms")
    
    print()


def demo_real_world_scenario():
    """Demo 4: Real-world vector database scenario."""
    print("=" * 70)
    print("Demo 4: Real-World Vector Database Scenario")
    print("=" * 70)
    
    d = 384  # e.g., sentence-transformers embedding size
    
    # Simulate a document database with updates
    print("\nSimulating a document database with continuous updates...")
    
    index = HNSWIDMapWithDeletion(d, M=32)
    
    # Day 1: Initial load
    print("\nDay 1: Loading 10000 documents...")
    batch1 = np.random.random((10000, d)).astype('float32')
    ids1 = list(range(1, 10001))
    index.add_with_ids(batch1, ids1)
    print(f"   Index size: {index.index.ntotal} vectors")
    
    # Day 2: Add new documents, delete old ones
    print("\nDay 2: Adding 1000 new docs, deleting 500 old docs...")
    batch2 = np.random.random((1000, d)).astype('float32')
    ids2 = list(range(10001, 11001))
    index.add_with_ids(batch2, ids2)
    
    # Delete some old documents
    ids_to_delete = list(range(1, 501))
    index.remove(ids_to_delete)
    print(f"   Index size: {index.index.ntotal} vectors")
    print(f"   Deleted: {len(index.deleted_external_ids)} vectors")
    
    # Day 3-7: Continue operations
    for day in range(3, 8):
        print(f"\nDay {day}: Adding 500 new docs, deleting 300 old docs...")
        batch = np.random.random((500, d)).astype('float32')
        start_id = 10000 + (day - 2) * 1000 + 1
        ids = list(range(start_id, start_id + 500))
        index.add_with_ids(batch, ids)
        
        # Delete some
        delete_start = 500 + (day - 3) * 300 + 1
        ids_to_delete = list(range(delete_start, delete_start + 300))
        index.remove(ids_to_delete)
        
        deleted_ratio = len(index.deleted_external_ids) / index.index.ntotal
        print(f"   Index size: {index.index.ntotal} vectors "
              f"({len(index.deleted_external_ids)} deleted, {deleted_ratio:.1%} ratio)")
    
    # Weekend: Consolidation
    print("\nWeekend: Performing maintenance (consolidation)...")
    index.consolidate()
    
    # Test search
    print("\nPerforming test search...")
    query = np.random.random((1, d)).astype('float32')
    distances, labels = index.search(query, k=10)
    print(f"   Search results: {labels[0]}")
    print(f"   Final index size: {index.index.ntotal} vectors")
    
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FAISS HNSW with Deletion Support - Complete Demo")
    print("=" * 70)
    print("\nThis demo shows how to implement vector deletion in FAISS HNSW")
    print("for building a production-ready Vector Database.\n")
    
    demo_basic_deletion()
    demo_custom_ids()
    demo_performance()
    demo_real_world_scenario()
    
    print("=" * 70)
    print("Summary and Recommendations")
    print("=" * 70)
    print("""
1. Lazy Deletion (Tombstone):
   ✓ Fast deletion (O(1))
   ✓ No rebuild needed
   ✓ Slight search overhead (5-15%)
   → Best for real-time systems

2. Periodic Consolidation:
   ✓ Reclaims memory
   ✓ Restores full speed
   ✓ Can be done offline
   → Run when deletion ratio > 20-30%

3. Custom ID Mapping:
   ✓ Use external IDs (database keys)
   ✓ Easier integration
   → Best for production VectorDB

4. Performance Tips:
   - Set appropriate efSearch (16-32 for speed, 64-128 for accuracy)
   - Consolidate during low-traffic periods
   - Monitor deletion ratio
   - Use batch operations when possible

5. When NOT to use HNSW:
   - Very high deletion rate (>50%)
   → Consider IndexIVFFlat instead
   - Need exact deletion immediately
   → Use IndexFlat or IndexIVFFlat
    """)
    
    print("\nDemo completed! Use these patterns in your VectorDB project.\n")
