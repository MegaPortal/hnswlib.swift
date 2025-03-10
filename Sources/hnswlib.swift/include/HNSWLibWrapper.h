#ifndef HNSWLIB_WRAPPER_H
#define HNSWLIB_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h> // For size_t

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types to represent the C++ objects
typedef struct HNSWIndex HNSWIndex;
typedef struct BFIndex BFIndex;

// Space types
typedef enum {
    SpaceTypeL2 = 0,
    SpaceTypeIP = 1,      // Inner product
    SpaceTypeCosine = 2   // Cosine similarity
} SpaceType;

// Creating and destroying indices
HNSWIndex* hnswlib_index_create(SpaceType space_type, int dim);
void hnswlib_index_free(HNSWIndex* index);

// Initialize the index
bool hnswlib_index_init(HNSWIndex* index, size_t max_elements, size_t M, size_t ef_construction, size_t random_seed, bool allow_replace_deleted);

// Add items
bool hnswlib_index_add_items(HNSWIndex* index, const float* data, size_t rows, size_t dim, const uint64_t* ids, int num_threads, bool replace_deleted);

// Search
bool hnswlib_index_search_knn(HNSWIndex* index, const float* query, size_t k, uint64_t* result_labels, float* result_distances, size_t query_count, int num_threads);

// Set ef parameter (search accuracy vs speed)
void hnswlib_index_set_ef(HNSWIndex* index, size_t ef);

// Get current parameters
size_t hnswlib_index_get_current_count(HNSWIndex* index);
size_t hnswlib_index_get_max_elements(HNSWIndex* index);
size_t hnswlib_index_get_ef(HNSWIndex* index);
size_t hnswlib_index_get_m(HNSWIndex* index);

// Save/load index
bool hnswlib_index_save(HNSWIndex* index, const char* path);
HNSWIndex* hnswlib_index_load(SpaceType space_type, int dim, const char* path, size_t max_elements, bool allow_replace_deleted);

// Mark/unmark deleted
void hnswlib_index_mark_deleted(HNSWIndex* index, uint64_t label);
void hnswlib_index_unmark_deleted(HNSWIndex* index, uint64_t label);

// Resize index
bool hnswlib_index_resize(HNSWIndex* index, size_t new_size);

// BruteForce index functions
BFIndex* hnswlib_bf_index_create(SpaceType space_type, int dim);
void hnswlib_bf_index_free(BFIndex* index);
bool hnswlib_bf_index_init(BFIndex* index, size_t max_elements);
bool hnswlib_bf_index_add_items(BFIndex* index, const float* data, size_t rows, size_t dim, const uint64_t* ids);
bool hnswlib_bf_index_search_knn(BFIndex* index, const float* query, size_t k, uint64_t* result_labels, float* result_distances, size_t query_count, int num_threads);

#ifdef __cplusplus
}
#endif

#endif // HNSWLIB_WRAPPER_H 