/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Demo: HNSW Index with Deletion Support
 * 
 * This demo shows how to implement vector deletion functionality for HNSW index.
 * 
 * HNSW doesn't natively support deletion, so we implement three strategies:
 * 1. Lazy deletion with tombstone marking (recommended for most cases)
 * 2. Rebuild from scratch (simple but expensive)
 * 3. Hybrid approach with periodic consolidation
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <unordered_set>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>

using namespace faiss;

//=============================================================================
// Strategy 1: Lazy Deletion with Tombstone Marking (IDSelector approach)
//=============================================================================

/**
 * Custom ID selector that filters out deleted IDs during search.
 * This is the most efficient approach for HNSW.
 */
class TombstoneIDSelector : public IDSelector {
public:
    std::unordered_set<idx_t> deleted_ids;
    
    TombstoneIDSelector() {}
    
    void mark_deleted(idx_t id) {
        deleted_ids.insert(id);
    }
    
    void mark_deleted_batch(const std::vector<idx_t>& ids) {
        deleted_ids.insert(ids.begin(), ids.end());
    }
    
    bool is_member(idx_t id) const override {
        return deleted_ids.find(id) == deleted_ids.end();
    }
    
    size_t get_deleted_count() const {
        return deleted_ids.size();
    }
    
    void clear() {
        deleted_ids.clear();
    }
};

/**
 * Enhanced HNSW index wrapper with deletion support
 */
class IndexHNSWWithDeletion {
public:
    std::unique_ptr<IndexHNSW> index;
    TombstoneIDSelector tombstone_selector;
    idx_t total_added = 0;
    
    // Threshold for triggering rebuild (when deleted ratio exceeds this)
    float rebuild_threshold = 0.3;
    
    IndexHNSWWithDeletion(int d, int M = 32, MetricType metric = METRIC_L2) {
        index = std::make_unique<IndexHNSW>(d, M, metric);
    }
    
    void train(idx_t n, const float* x) {
        index->train(n, x);
    }
    
    void add(idx_t n, const float* x) {
        index->add(n, x);
        total_added += n;
    }
    
    /**
     * Mark vectors as deleted (lazy deletion)
     */
    size_t remove_ids(const std::vector<idx_t>& ids_to_remove) {
        tombstone_selector.mark_deleted_batch(ids_to_remove);
        
        // Check if we should trigger a rebuild
        float deleted_ratio = (float)tombstone_selector.get_deleted_count() / 
                             (float)index->ntotal;
        
        if (deleted_ratio > rebuild_threshold) {
            printf("Deleted ratio %.2f%% exceeds threshold, consider calling consolidate()\n",
                   deleted_ratio * 100);
        }
        
        return ids_to_remove.size();
    }
    
    /**
     * Search with tombstone filtering
     */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) {
        
        // We need to fetch more results than k to account for deleted ones
        size_t deleted_count = tombstone_selector.get_deleted_count();
        float deleted_ratio = (float)deleted_count / (float)index->ntotal;
        
        // Calculate how many extra results to fetch
        idx_t k_search = k;
        if (deleted_ratio > 0.01) {
            // Fetch extra results based on deletion ratio
            k_search = std::min(
                (idx_t)(k * (1.0 + deleted_ratio * 2)),
                index->ntotal
            );
            k_search = std::max(k_search, k + 10);
        }

        std::vector<float> temp_distances(n * k_search);
        std::vector<idx_t> temp_labels(n * k_search);
        
        // Perform search
        index->search(n, x, k_search, temp_distances.data(), temp_labels.data());

        // Filter out deleted IDs
        for (idx_t i = 0; i < n; i++) {
            idx_t valid_count = 0;
            for (idx_t j = 0; j < k_search && valid_count < k; j++) {
                idx_t label = temp_labels[i * k_search + j];
                if (label >= 0 && tombstone_selector.is_member(label)) {
                    distances[i * k + valid_count] = temp_distances[i * k_search + j];
                    labels[i * k + valid_count] = label;
                    valid_count++;
                }
            }
            // Fill remaining with -1 if not enough valid results
            for (idx_t j = valid_count; j < k; j++) {
                distances[i * k + j] = INFINITY;
                labels[i * k + j] = -1;
            }
        }
    }
    
    /**
     * Consolidate: rebuild index without deleted vectors
     * This should be called periodically when deletion ratio is high
     */
    void consolidate(const float* all_vectors = nullptr) {
        if (tombstone_selector.get_deleted_count() == 0) {
            return; // Nothing to consolidate
        }
        
        printf("Consolidating index: removing %zu deleted vectors from %lld total\n",
               tombstone_selector.get_deleted_count(), (long long)index->ntotal);
        
        // Extract valid vectors
        std::vector<float> valid_vectors;
        std::vector<idx_t> valid_ids;
        valid_vectors.reserve(index->d * (index->ntotal - tombstone_selector.get_deleted_count()));
        
        if (all_vectors != nullptr) {
            // If we have access to original vectors, use them
            for (idx_t i = 0; i < index->ntotal; i++) {
                if (tombstone_selector.is_member(i)) {
                    valid_ids.push_back(i);
                    for (int j = 0; j < index->d; j++) {
                        valid_vectors.push_back(all_vectors[i * index->d + j]);
                    }
                }
            }
        } else {
            // Reconstruct from index (if supported)
            for (idx_t i = 0; i < index->ntotal; i++) {
                if (tombstone_selector.is_member(i)) {
                    valid_ids.push_back(i);
                    std::vector<float> vec(index->d);
                    index->reconstruct(i, vec.data());
                    valid_vectors.insert(valid_vectors.end(), vec.begin(), vec.end());
                }
            }
        }
        
        // Create new index
        int d = index->d;
        int M = index->hnsw.cum_nneighbor_per_level[1] / 2; // Recover M parameter
        MetricType metric = index->metric_type;
        
        auto new_index = std::make_unique<IndexHNSW>(d, M, metric);
        new_index->hnsw.efConstruction = index->hnsw.efConstruction;
        new_index->hnsw.efSearch = index->hnsw.efSearch;
        
        // Add valid vectors to new index
        if (!valid_vectors.empty()) {
            new_index->add(valid_ids.size(), valid_vectors.data());
        }
        
        // Replace old index
        index = std::move(new_index);
        tombstone_selector.clear();
        
        printf("Consolidation complete: index now has %lld vectors\n", 
               (long long)index->ntotal);
    }
    
    idx_t ntotal() const {
        return index->ntotal - tombstone_selector.get_deleted_count();
    }
    
    idx_t ntotal_with_deleted() const {
        return index->ntotal;
    }
    
    void reset() {
        index->reset();
        tombstone_selector.clear();
        total_added = 0;
    }
};

//=============================================================================
// Strategy 2: Hybrid with IndexIDMap
//=============================================================================

/**
 * Use IndexIDMap to manage custom IDs, making deletion easier to track
 */
class IndexHNSWIDMapWithDeletion {
public:
    std::unique_ptr<IndexHNSW> hnsw_index;
    std::vector<idx_t> id_map;  // Maps internal index to external IDs
    std::unordered_set<idx_t> deleted_external_ids;
    int d;
    
    IndexHNSWIDMapWithDeletion(int d, int M = 32, MetricType metric = METRIC_L2)
        : d(d) {
        hnsw_index = std::make_unique<IndexHNSW>(d, M, metric);
    }
    
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) {
        hnsw_index->add(n, x);
        for (idx_t i = 0; i < n; i++) {
            id_map.push_back(xids[i]);
        }
    }
    
    size_t remove_ids(const std::vector<idx_t>& external_ids) {
        deleted_external_ids.insert(external_ids.begin(), external_ids.end());
        return external_ids.size();
    }
    
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) {
        
        // Calculate how many to fetch accounting for deletions
        float deleted_ratio = (float)deleted_external_ids.size() / 
                             (float)hnsw_index->ntotal;
        idx_t k_search = std::max(k * 2, k + 10);
        if (deleted_ratio > 0.1) {
            k_search = std::min((idx_t)(k * 3), hnsw_index->ntotal);
        }
        
        std::vector<float> temp_distances(n * k_search);
        std::vector<idx_t> temp_labels(n * k_search);
        
        hnsw_index->search(n, x, k_search, temp_distances.data(), temp_labels.data());
        
        // Map internal IDs to external IDs and filter deleted
        for (idx_t i = 0; i < n; i++) {
            idx_t valid_count = 0;
            for (idx_t j = 0; j < k_search && valid_count < k; j++) {
                idx_t internal_id = temp_labels[i * k_search + j];
                if (internal_id >= 0 && internal_id < id_map.size()) {
                    idx_t external_id = id_map[internal_id];
                    if (deleted_external_ids.find(external_id) == deleted_external_ids.end()) {
                        distances[i * k + valid_count] = temp_distances[i * k_search + j];
                        labels[i * k + valid_count] = external_id;
                        valid_count++;
                    }
                }
            }
            for (idx_t j = valid_count; j < k; j++) {
                distances[i * k + j] = INFINITY;
                labels[i * k + j] = -1;
            }
        }
    }
    
    idx_t ntotal() const {
        return hnsw_index->ntotal - deleted_external_ids.size();
    }
};

//=============================================================================
// Demo and Testing
//=============================================================================

int main() {
    int d = 128;      // dimension
    int nb = 10000;   // database size
    int nq = 100;     // number of queries
    
    printf("=== HNSW with Deletion Support Demo ===\n\n");
    
    // Generate random data
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    srand(time(nullptr));
    for (int i = 0; i < nb * d; i++) {
        xb[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < nq * d; i++) {
        xq[i] = (float)rand() / RAND_MAX;
    }
    
    //-------------------------------------------------------------------------
    // Test Strategy 1: Lazy Deletion with Tombstone
    //-------------------------------------------------------------------------
    printf("Strategy 1: Lazy Deletion with Tombstone\n");
    printf("=========================================\n\n");
    
    IndexHNSWWithDeletion index_with_deletion(d, 32);
    
    // Build index
    printf("Adding %d vectors...\n", nb);
    index_with_deletion.add(nb, xb.data());
    printf("Index built with %lld vectors\n\n", (long long)index_with_deletion.ntotal());
    
    // Initial search
    int k = 10;
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    
    printf("Performing initial search for %d queries (k=%d)...\n", nq, k);
    index_with_deletion.search(nq, xq.data(), k, distances.data(), labels.data());
    printf("First query results: ");
    for (int i = 0; i < 5; i++) {
        printf("%lld ", (long long)labels[i]);
    }
    printf("\n\n");
    
    // Delete some vectors
    printf("Deleting 1000 vectors...\n");
    std::vector<idx_t> ids_to_delete;
    for (int i = 0; i < 1000; i++) {
        ids_to_delete.push_back(i * 5); // Delete every 5th vector starting from 0
    }
    index_with_deletion.remove_ids(ids_to_delete);
    printf("Deleted %zu vectors. Active vectors: %lld (Total with tombstones: %lld)\n\n",
           ids_to_delete.size(),
           (long long)index_with_deletion.ntotal(),
           (long long)index_with_deletion.ntotal_with_deleted());
    
    // Search after deletion
    printf("Searching after deletion...\n");
    index_with_deletion.search(nq, xq.data(), k, distances.data(), labels.data());
    printf("First query results after deletion: ");
    for (int i = 0; i < 5; i++) {
        printf("%lld ", (long long)labels[i]);
    }
    printf("\n");
    
    // Verify deleted IDs are not in results
    bool found_deleted = false;
    for (int i = 0; i < nq * k; i++) {
        if (labels[i] >= 0) {
            for (auto deleted_id : ids_to_delete) {
                if (labels[i] == deleted_id) {
                    found_deleted = true;
                    break;
                }
            }
        }
    }
    printf("Deleted IDs in results: %s\n\n", found_deleted ? "YES (ERROR!)" : "NO (Correct!)");
    
    // Delete more vectors to trigger consolidation warning
    printf("Deleting 2000 more vectors (30%% threshold)...\n");
    ids_to_delete.clear();
    for (int i = 1000; i < 3000; i++) {
        ids_to_delete.push_back(i);
    }
    index_with_deletion.remove_ids(ids_to_delete);
    printf("\n");
    
    // Consolidate
    printf("Consolidating index to remove tombstones...\n");
    index_with_deletion.consolidate(xb.data());
    printf("After consolidation: %lld active vectors\n\n", 
           (long long)index_with_deletion.ntotal());
    
    //-------------------------------------------------------------------------
    // Test Strategy 2: With Custom IDs
    //-------------------------------------------------------------------------
    printf("\nStrategy 2: IndexIDMap approach with Custom IDs\n");
    printf("================================================\n\n");
    
    IndexHNSWIDMapWithDeletion index_with_idmap(d, 32);
    
    // Generate custom IDs (e.g., using external database keys)
    std::vector<idx_t> custom_ids(nb);
    for (int i = 0; i < nb; i++) {
        custom_ids[i] = 1000000 + i; // Start from 1000000
    }
    
    printf("Adding %d vectors with custom IDs...\n", nb);
    index_with_idmap.add_with_ids(nb, xb.data(), custom_ids.data());
    printf("Index built with %lld vectors\n\n", (long long)index_with_idmap.ntotal());
    
    // Search with custom IDs
    printf("Searching with custom IDs...\n");
    index_with_idmap.search(nq, xq.data(), k, distances.data(), labels.data());
    printf("First query results (custom IDs): ");
    for (int i = 0; i < 5; i++) {
        printf("%lld ", (long long)labels[i]);
    }
    printf("\n\n");
    
    // Delete using custom IDs
    printf("Deleting vectors with custom IDs 1000000-1000099...\n");
    std::vector<idx_t> custom_ids_to_delete;
    for (int i = 0; i < 100; i++) {
        custom_ids_to_delete.push_back(1000000 + i);
    }
    index_with_idmap.remove_ids(custom_ids_to_delete);
    printf("Active vectors: %lld\n\n", (long long)index_with_idmap.ntotal());
    
    // Search after deletion
    printf("Searching after deletion...\n");
    index_with_idmap.search(nq, xq.data(), k, distances.data(), labels.data());
    printf("First query results after deletion: ");
    for (int i = 0; i < 5; i++) {
        printf("%lld ", (long long)labels[i]);
    }
    printf("\n\n");
    
    //-------------------------------------------------------------------------
    // Performance comparison
    //-------------------------------------------------------------------------
    printf("Performance Tips:\n");
    printf("==================\n");
    printf("1. Use lazy deletion for real-time operations (no rebuild cost)\n");
    printf("2. Periodically consolidate when deletion ratio > 20-30%%\n");
    printf("3. For write-heavy workloads, consider using IndexIVFFlat instead\n");
    printf("4. Adjust k_search multiplier based on your deletion ratio\n");
    printf("5. Use IndexIDMap approach if you need custom external IDs\n\n");
    
    printf("Demo completed successfully!\n");
    
    return 0;
}
