#include "HNSWLibWrapper.h"
#include "../hnswlib.cpp/hnswlib.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

using namespace hnswlib;

// Helper function for parallel processing
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

// Helper for vector normalization (used for cosine similarity)
inline void normalize_vector(float* data, float* norm_array, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++)
        norm += data[i] * data[i];
    norm = 1.0f / (sqrtf(norm) + 1e-30f);
    for (int i = 0; i < dim; i++)
        norm_array[i] = data[i] * norm;
}

// HNSW Index implementation
struct HNSWIndex {
    SpaceType space_type;
    int dim;
    bool normalize;
    bool index_inited;
    bool ep_added;
    int num_threads_default;
    labeltype cur_l;
    HierarchicalNSW<float>* appr_alg;
    SpaceInterface<float>* space;
    size_t default_ef;
    
    HNSWIndex(SpaceType space_type, int dim) 
        : space_type(space_type), 
          dim(dim), 
          normalize(false), 
          index_inited(false), 
          ep_added(false), 
          num_threads_default(std::thread::hardware_concurrency()),
          cur_l(0),
          appr_alg(nullptr),
          space(nullptr),
          default_ef(10) {
        
        if (space_type == SpaceTypeL2) {
            space = new L2Space(dim);
        } else if (space_type == SpaceTypeIP) {
            space = new InnerProductSpace(dim);
        } else if (space_type == SpaceTypeCosine) {
            space = new InnerProductSpace(dim);
            normalize = true;
        }
    }
    
    ~HNSWIndex() {
        if (space) {
            delete space;
        }
        if (appr_alg) {
            delete appr_alg;
        }
    }
};

// BruteForce Index implementation
struct BFIndex {
    SpaceType space_type;
    int dim;
    bool normalize;
    int num_threads_default;
    labeltype cur_l;
    BruteforceSearch<float>* alg;
    SpaceInterface<float>* space;
    
    BFIndex(SpaceType space_type, int dim) 
        : space_type(space_type), 
          dim(dim), 
          normalize(false), 
          num_threads_default(std::thread::hardware_concurrency()),
          cur_l(0),
          alg(nullptr),
          space(nullptr) {
        
        if (space_type == SpaceTypeL2) {
            space = new L2Space(dim);
        } else if (space_type == SpaceTypeIP) {
            space = new InnerProductSpace(dim);
        } else if (space_type == SpaceTypeCosine) {
            space = new InnerProductSpace(dim);
            normalize = true;
        }
    }
    
    ~BFIndex() {
        if (space) {
            delete space;
        }
        if (alg) {
            delete alg;
        }
    }
};

// HNSW Index Functions
extern "C" {

HNSWIndex* hnswlib_index_create(SpaceType space_type, int dim) {
    try {
        return new HNSWIndex(space_type, dim);
    } catch (const std::exception& e) {
        std::cerr << "Error creating index: " << e.what() << std::endl;
        return nullptr;
    }
}

void hnswlib_index_free(HNSWIndex* index) {
    if (index) {
        delete index;
    }
}

bool hnswlib_index_init(HNSWIndex* index, size_t max_elements, size_t M, size_t ef_construction, size_t random_seed, bool allow_replace_deleted) {
    if (!index || !index->space) return false;
    
    try {
        if (index->appr_alg) {
            delete index->appr_alg;
        }
        
        index->cur_l = 0;
        index->appr_alg = new HierarchicalNSW<float>(index->space, max_elements, M, ef_construction, random_seed, allow_replace_deleted);
        index->index_inited = true;
        index->ep_added = false;
        index->appr_alg->ef_ = index->default_ef;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing index: " << e.what() << std::endl;
        return false;
    }
}

bool hnswlib_index_add_items(HNSWIndex* index, const float* data, size_t rows, size_t dim, const uint64_t* ids, int num_threads, bool replace_deleted) {
    if (!index || !index->appr_alg || dim != (size_t)index->dim) return false;
    
    try {
        if (num_threads <= 0) {
            num_threads = index->num_threads_default;
        }
        
        // Avoid using threads when the number of additions is small
        if (rows <= (size_t)(num_threads * 4)) {
            num_threads = 1;
        }
        
        int start = 0;
        if (!index->ep_added) {
            size_t id = ids ? ids[0] : index->cur_l;
            float* vector_data = const_cast<float*>(&data[0]);
            std::vector<float> norm_array(index->dim);
            
            if (index->normalize) {
                normalize_vector(vector_data, norm_array.data(), index->dim);
                vector_data = norm_array.data();
            }
            
            index->appr_alg->addPoint(vector_data, id, replace_deleted);
            start = 1;
            index->ep_added = true;
        }
        
        if (index->normalize == false) {
            ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                size_t id = ids ? ids[row] : (index->cur_l + row);
                index->appr_alg->addPoint(&data[row * dim], id, replace_deleted);
            });
        } else {
            std::vector<float> norm_array(num_threads * index->dim);
            ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                // Normalize vector
                size_t start_idx = threadId * index->dim;
                normalize_vector(const_cast<float*>(&data[row * dim]), &norm_array[start_idx], index->dim);
                
                size_t id = ids ? ids[row] : (index->cur_l + row);
                index->appr_alg->addPoint(&norm_array[start_idx], id, replace_deleted);
            });
        }
        
        index->cur_l += rows;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error adding items: " << e.what() << std::endl;
        return false;
    }
}

bool hnswlib_index_search_knn(HNSWIndex* index, const float* query, size_t k, uint64_t* result_labels, float* result_distances, size_t query_count, int num_threads) {
    if (!index || !index->appr_alg) return false;
    
    try {
        if (num_threads <= 0) {
            num_threads = index->num_threads_default;
        }
        
        // Avoid using threads when the number of searches is small
        if (query_count <= (size_t)(num_threads * 4)) {
            num_threads = 1;
        }
        
        if (!index->normalize) {
            ParallelFor(0, query_count, num_threads, [&](size_t i, size_t threadId) {
                std::priority_queue<std::pair<float, labeltype>> result = 
                    index->appr_alg->searchKnn(&query[i * index->dim], k);
                
                if (result.size() != k) {
                    throw std::runtime_error("Cannot return results. Probably ef or M is too small");
                }
                
                for (int j = k - 1; j >= 0; j--) {
                    auto& result_tuple = result.top();
                    result_distances[i * k + j] = result_tuple.first;
                    result_labels[i * k + j] = result_tuple.second;
                    result.pop();
                }
            });
        } else {
            std::vector<float> norm_array(num_threads * index->dim);
            ParallelFor(0, query_count, num_threads, [&](size_t i, size_t threadId) {
                size_t start_idx = threadId * index->dim;
                normalize_vector(const_cast<float*>(&query[i * index->dim]), &norm_array[start_idx], index->dim);
                
                std::priority_queue<std::pair<float, labeltype>> result = 
                    index->appr_alg->searchKnn(&norm_array[start_idx], k);
                
                if (result.size() != k) {
                    throw std::runtime_error("Cannot return results. Probably ef or M is too small");
                }
                
                for (int j = k - 1; j >= 0; j--) {
                    auto& result_tuple = result.top();
                    result_distances[i * k + j] = result_tuple.first;
                    result_labels[i * k + j] = result_tuple.second;
                    result.pop();
                }
            });
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error searching: " << e.what() << std::endl;
        return false;
    }
}

void hnswlib_index_set_ef(HNSWIndex* index, size_t ef) {
    if (!index) return;
    
    index->default_ef = ef;
    if (index->appr_alg) {
        index->appr_alg->ef_ = ef;
    }
}

size_t hnswlib_index_get_current_count(HNSWIndex* index) {
    if (!index || !index->appr_alg) return 0;
    return index->appr_alg->cur_element_count;
}

size_t hnswlib_index_get_max_elements(HNSWIndex* index) {
    if (!index || !index->appr_alg) return 0;
    return index->appr_alg->max_elements_;
}

size_t hnswlib_index_get_ef(HNSWIndex* index) {
    if (!index || !index->appr_alg) return 0;
    return index->appr_alg->ef_;
}

size_t hnswlib_index_get_m(HNSWIndex* index) {
    if (!index || !index->appr_alg) return 0;
    return index->appr_alg->M_;
}

bool hnswlib_index_save(HNSWIndex* index, const char* path) {
    if (!index || !index->appr_alg) return false;
    
    try {
        index->appr_alg->saveIndex(path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving index: " << e.what() << std::endl;
        return false;
    }
}

HNSWIndex* hnswlib_index_load(SpaceType space_type, int dim, const char* path, size_t max_elements, bool allow_replace_deleted) {
    try {
        HNSWIndex* index = new HNSWIndex(space_type, dim);
        if (!index->space) {
            delete index;
            return nullptr;
        }
        
        index->appr_alg = new HierarchicalNSW<float>(index->space, path, false, max_elements, allow_replace_deleted);
        index->cur_l = index->appr_alg->cur_element_count;
        index->index_inited = true;
        index->ep_added = true;
        
        return index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        return nullptr;
    }
}

void hnswlib_index_mark_deleted(HNSWIndex* index, uint64_t label) {
    if (!index || !index->appr_alg) return;
    
    try {
        index->appr_alg->markDelete(label);
    } catch (const std::exception& e) {
        std::cerr << "Error marking item as deleted: " << e.what() << std::endl;
    }
}

void hnswlib_index_unmark_deleted(HNSWIndex* index, uint64_t label) {
    if (!index || !index->appr_alg) return;
    
    try {
        index->appr_alg->unmarkDelete(label);
    } catch (const std::exception& e) {
        std::cerr << "Error unmarking deletion: " << e.what() << std::endl;
    }
}

bool hnswlib_index_resize(HNSWIndex* index, size_t new_size) {
    if (!index || !index->appr_alg) return false;
    
    try {
        index->appr_alg->resizeIndex(new_size);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error resizing index: " << e.what() << std::endl;
        return false;
    }
}

// BruteForce Index Functions
BFIndex* hnswlib_bf_index_create(SpaceType space_type, int dim) {
    try {
        return new BFIndex(space_type, dim);
    } catch (const std::exception& e) {
        std::cerr << "Error creating BF index: " << e.what() << std::endl;
        return nullptr;
    }
}

void hnswlib_bf_index_free(BFIndex* index) {
    if (index) {
        delete index;
    }
}

bool hnswlib_bf_index_init(BFIndex* index, size_t max_elements) {
    if (!index || !index->space) return false;
    
    try {
        if (index->alg) {
            delete index->alg;
        }
        
        index->cur_l = 0;
        index->alg = new BruteforceSearch<float>(index->space, max_elements);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing BF index: " << e.what() << std::endl;
        return false;
    }
}

bool hnswlib_bf_index_add_items(BFIndex* index, const float* data, size_t rows, size_t dim, const uint64_t* ids) {
    if (!index || !index->alg || dim != (size_t)index->dim) return false;
    
    try {
        for (size_t row = 0; row < rows; row++) {
            size_t id = ids ? ids[row] : index->cur_l + row;
            
            if (!index->normalize) {
                index->alg->addPoint(&data[row * dim], id);
            } else {
                std::vector<float> normalized_vector(index->dim);
                normalize_vector(const_cast<float*>(&data[row * dim]), normalized_vector.data(), index->dim);
                index->alg->addPoint(normalized_vector.data(), id);
            }
        }
        
        index->cur_l += rows;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error adding items to BF index: " << e.what() << std::endl;
        return false;
    }
}

bool hnswlib_bf_index_search_knn(BFIndex* index, const float* query, size_t k, uint64_t* result_labels, float* result_distances, size_t query_count, int num_threads) {
    if (!index || !index->alg) return false;
    
    try {
        if (num_threads <= 0) {
            num_threads = index->num_threads_default;
        }
        
        ParallelFor(0, query_count, num_threads, [&](size_t i, size_t threadId) {
            std::priority_queue<std::pair<float, labeltype>> result;
            
            if (!index->normalize) {
                result = index->alg->searchKnn(&query[i * index->dim], k);
            } else {
                std::vector<float> normalized_query(index->dim);
                normalize_vector(const_cast<float*>(&query[i * index->dim]), normalized_query.data(), index->dim);
                result = index->alg->searchKnn(normalized_query.data(), k);
            }
            
            for (int j = k - 1; j >= 0; j--) {
                auto& result_tuple = result.top();
                result_distances[i * k + j] = result_tuple.first;
                result_labels[i * k + j] = result_tuple.second;
                result.pop();
            }
        });
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error searching BF index: " << e.what() << std::endl;
        return false;
    }
}

} // extern "C" 