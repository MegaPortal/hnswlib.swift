// The Swift Programming Language
// https://docs.swift.org/swift-book

// hnswlib.swift - Swift bindings for HNSW library
// HNSW (Hierarchical Navigable Small World) is a fast approximate nearest neighbor search library

import Foundation
// Import the C functions from our wrapper
import hnswlib_c
// Import for C standard library functions like strdup and free
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Type of distance metric to use for the index
public enum SpaceType: Int32 {
    /// Euclidean distance (L2)
    case l2 = 0
    /// Inner product (dot product), useful for cosine similarity with normalized vectors
    case innerProduct = 1
    /// Cosine similarity (automatically normalizes vectors)
    case cosine = 2
}

/// Error types that can be thrown by HNSW operations
public enum HNSWError: Error {
    case initializationFailed
    case outOfMemory
    case addItemsFailed
    case searchFailed
    case invalidDimension
    case saveFailed
    case loadFailed
    case resizeFailed
}

/// Main class for the HNSW index
public class HNSWIndex {
    private var indexPtr: OpaquePointer?
    
    /// The dimension of the vectors in the index
    public let dim: Int
    
    /// The space type (L2, inner product, cosine)
    public let spaceType: SpaceType
    
    /// Creates a new index
    /// - Parameters:
    ///   - spaceType: The distance metric to use
    ///   - dim: The dimension of vectors to index
    public init(spaceType: SpaceType, dim: Int) throws {
        self.spaceType = spaceType
        self.dim = dim
        
        guard let indexPtr = hnswlib_index_create(spaceType.rawValue, Int32(dim)) else {
            throw HNSWError.initializationFailed
        }
        
        self.indexPtr = indexPtr
    }
    
    deinit {
        if let indexPtr = indexPtr {
            hnswlib_index_free(indexPtr)
        }
    }
    
    /// Initialize the index with parameters
    /// - Parameters:
    ///   - maxElements: Maximum number of elements the index can hold
    ///   - m: Number of bidirectional links created for each element during construction
    ///   - efConstruction: Size of the dynamic list for the nearest neighbors during construction
    ///   - randomSeed: Seed for the random number generator
    ///   - allowReplaceDeleted: Whether to allow replacing deleted elements
    public func initIndex(maxElements: Int, m: Int = 16, efConstruction: Int = 200, randomSeed: UInt = 100, allowReplaceDeleted: Bool = false) throws {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        if !hnswlib_index_init(indexPtr, size_t(maxElements), size_t(m), size_t(efConstruction), size_t(randomSeed), allowReplaceDeleted) {
            throw HNSWError.initializationFailed
        }
    }
    
    /// Add items to the index
    /// - Parameters:
    ///   - data: The vectors to add, should be a 2D array of dimension [n, dim]
    ///   - ids: Optional array of item IDs, if nil, sequential IDs will be assigned
    ///   - numThreads: Number of threads to use for parallel insertion, -1 for auto
    ///   - replaceDeleted: Whether to replace deleted elements
    public func addItems(data: [[Float]], ids: [UInt64]? = nil, numThreads: Int = -1, replaceDeleted: Bool = false) throws {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        let rows = data.count
        guard rows > 0 else { return }
        
        let vecDim = data[0].count
        guard vecDim == dim else {
            throw HNSWError.invalidDimension
        }
        
        // Flatten the 2D array
        var flattenedData = [Float](repeating: 0, count: rows * dim)
        for i in 0..<rows {
            for j in 0..<dim {
                flattenedData[i * dim + j] = data[i][j]
            }
        }
        
        var idsArray: [UInt64]?
        if let ids = ids {
            guard ids.count == rows else {
                throw HNSWError.addItemsFailed
            }
            idsArray = ids
        }
        
        if !hnswlib_index_add_items(indexPtr, flattenedData, size_t(rows), size_t(dim), idsArray, Int32(numThreads), replaceDeleted) {
            throw HNSWError.addItemsFailed
        }
    }
    
    /// Search for k nearest neighbors
    /// - Parameters:
    ///   - query: The query vectors, should be a 2D array of dimension [n, dim]
    ///   - k: Number of nearest neighbors to return
    ///   - numThreads: Number of threads to use for parallel search, -1 for auto
    /// - Returns: Tuple with (labels, distances) where both are 2D arrays of shape [n, k]
    public func searchKnn(query: [[Float]], k: Int, numThreads: Int = -1) throws -> (labels: [[UInt64]], distances: [[Float]]) {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        let queryCount = query.count
        guard queryCount > 0 else {
            return ([], [])
        }
        
        let vecDim = query[0].count
        guard vecDim == dim else {
            throw HNSWError.invalidDimension
        }
        
        // Flatten the 2D array
        var flattenedQuery = [Float](repeating: 0, count: queryCount * dim)
        for i in 0..<queryCount {
            for j in 0..<dim {
                flattenedQuery[i * dim + j] = query[i][j]
            }
        }
        
        // Allocate arrays for results
        var resultLabels = [UInt64](repeating: 0, count: queryCount * k)
        var resultDistances = [Float](repeating: 0, count: queryCount * k)
        
        if !hnswlib_index_search_knn(indexPtr, flattenedQuery, size_t(k), &resultLabels, &resultDistances, size_t(queryCount), Int32(numThreads)) {
            throw HNSWError.searchFailed
        }
        
        // Reshape results
        var labels = [[UInt64]](repeating: [UInt64](repeating: 0, count: k), count: queryCount)
        var distances = [[Float]](repeating: [Float](repeating: 0, count: k), count: queryCount)
        
        for i in 0..<queryCount {
            for j in 0..<k {
                labels[i][j] = resultLabels[i * k + j]
                distances[i][j] = resultDistances[i * k + j]
            }
        }
        
        return (labels, distances)
    }
    
    /// Set the ef parameter (search time accuracy vs. speed tradeoff)
    /// - Parameter ef: The size of the dynamic list for the nearest neighbors at search time
    public func setEf(ef: Int) {
        guard let indexPtr = indexPtr else { return }
        hnswlib_index_set_ef(indexPtr, size_t(ef))
    }
    
    /// Get current count of elements in the index
    public var currentCount: Int {
        guard let indexPtr = indexPtr else { return 0 }
        return Int(hnswlib_index_get_current_count(indexPtr))
    }
    
    /// Get maximum number of elements allowed in the index
    public var maxElements: Int {
        guard let indexPtr = indexPtr else { return 0 }
        return Int(hnswlib_index_get_max_elements(indexPtr))
    }
    
    /// Get current ef parameter value
    public var ef: Int {
        guard let indexPtr = indexPtr else { return 0 }
        return Int(hnswlib_index_get_ef(indexPtr))
    }
    
    /// Get current M parameter value
    public var m: Int {
        guard let indexPtr = indexPtr else { return 0 }
        return Int(hnswlib_index_get_m(indexPtr))
    }
    
    /// Save the index to a file
    /// - Parameter path: Path to save the index
    public func saveIndex(path: String) throws {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        // Validate path is not empty
        guard !path.isEmpty else {
            print("Error: Cannot save index to empty path")
            throw HNSWError.saveFailed
        }
        
        // We need to explicitly manage the string conversion to make sure it's retained 
        // throughout the C function call
        var result = false
        path.withCString { cString in
            // Create a copy of the C string to ensure it's stable and properly null-terminated
            guard let stableCString = strdup(cString) else {
                print("Error: Failed to allocate memory for path string")
                return
            }
            defer {
                // Free the duplicated string when we're done
                free(stableCString)
            }
            
            // Call the C function with our stable C string
            result = hnswlib_index_save(indexPtr, stableCString)
        }
        
        if !result {
            throw HNSWError.saveFailed
        }
    }
    
    /// Mark an item as deleted
    /// - Parameter label: ID of the item to mark as deleted
    public func markDeleted(label: UInt64) {
        guard let indexPtr = indexPtr else { return }
        hnswlib_index_mark_deleted(indexPtr, label)
    }
    
    /// Unmark a previously deleted item
    /// - Parameter label: ID of the item to unmark
    public func unmarkDeleted(label: UInt64) {
        guard let indexPtr = indexPtr else { return }
        hnswlib_index_unmark_deleted(indexPtr, label)
    }
    
    /// Resize the index to a new maximum size
    /// - Parameter newSize: New maximum number of elements
    public func resizeIndex(newSize: Int) throws {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        if !hnswlib_index_resize(indexPtr, size_t(newSize)) {
            throw HNSWError.resizeFailed
        }
    }
    
    /// Load an index from a file
    /// - Parameters:
    ///   - spaceType: Space type of the index
    ///   - dim: Dimensionality of the index
    ///   - path: Path to the index file
    ///   - maxElements: Maximum number of elements (0 to use the value from the file)
    ///   - allowReplaceDeleted: Whether deleted elements can be replaced
    /// - Returns: Loaded HNSW index
    public static func loadIndex(spaceType: SpaceType, dim: Int, path: String, maxElements: Int = 0, allowReplaceDeleted: Bool = false) throws -> HNSWIndex {
        // Validate path is not empty
        guard !path.isEmpty else {
            print("Error: Cannot load index from empty path")
            throw HNSWError.loadFailed
        }
        
        var resultPtr: OpaquePointer?
        
        // Use the path's withCString to handle string to C conversion properly
        path.withCString { cString in
            // Create a copy of the C string to ensure it's stable
            guard let stableCString = strdup(cString) else {
                print("Error: Failed to allocate memory for path string when loading index")
                return
            }
            defer {
                // Free the duplicated string when we're done
                free(stableCString)
            }
            
            resultPtr = hnswlib_index_load(spaceType.rawValue, Int32(dim), stableCString, size_t(maxElements), allowReplaceDeleted)
        }
        
        guard let indexPtr = resultPtr else {
            throw HNSWError.loadFailed
        }
        
        let index = try HNSWIndex(spaceType: spaceType, dim: dim)
        index.indexPtr = indexPtr
        return index
    }
}

/// BruteForce index for exact nearest neighbor search (for benchmarking and testing)
public class BFIndex {
    private var indexPtr: OpaquePointer?
    
    /// The dimension of the vectors in the index
    public let dim: Int
    
    /// The space type (L2, inner product, cosine)
    public let spaceType: SpaceType
    
    /// Creates a new BruteForce index
    /// - Parameters:
    ///   - spaceType: The distance metric to use
    ///   - dim: The dimension of vectors to index
    public init(spaceType: SpaceType, dim: Int) throws {
        self.spaceType = spaceType
        self.dim = dim
        
        guard let indexPtr = hnswlib_bf_index_create(spaceType.rawValue, Int32(dim)) else {
            throw HNSWError.initializationFailed
        }
        
        self.indexPtr = indexPtr
    }
    
    deinit {
        if let indexPtr = indexPtr {
            hnswlib_bf_index_free(indexPtr)
        }
    }
    
    /// Initialize the index with parameters
    /// - Parameter maxElements: Maximum number of elements the index can hold
    public func initIndex(maxElements: Int) throws {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        if !hnswlib_bf_index_init(indexPtr, size_t(maxElements)) {
            throw HNSWError.initializationFailed
        }
    }
    
    /// Add items to the index
    /// - Parameters:
    ///   - data: The vectors to add, should be a 2D array of dimension [n, dim]
    ///   - ids: Optional array of item IDs, if nil, sequential IDs will be assigned
    public func addItems(data: [[Float]], ids: [UInt64]? = nil) throws {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        let rows = data.count
        guard rows > 0 else { return }
        
        let vecDim = data[0].count
        guard vecDim == dim else {
            throw HNSWError.invalidDimension
        }
        
        // Flatten the 2D array
        var flattenedData = [Float](repeating: 0, count: rows * dim)
        for i in 0..<rows {
            for j in 0..<dim {
                flattenedData[i * dim + j] = data[i][j]
            }
        }
        
        var idsArray: [UInt64]?
        if let ids = ids {
            guard ids.count == rows else {
                throw HNSWError.addItemsFailed
            }
            idsArray = ids
        }
        
        if !hnswlib_bf_index_add_items(indexPtr, flattenedData, size_t(rows), size_t(dim), idsArray) {
            throw HNSWError.addItemsFailed
        }
    }
    
    /// Search for k nearest neighbors
    /// - Parameters:
    ///   - query: The query vectors, should be a 2D array of dimension [n, dim]
    ///   - k: Number of nearest neighbors to return
    ///   - numThreads: Number of threads to use for parallel search, -1 for auto
    /// - Returns: Tuple with (labels, distances) where both are 2D arrays of shape [n, k]
    public func searchKnn(query: [[Float]], k: Int, numThreads: Int = -1) throws -> (labels: [[UInt64]], distances: [[Float]]) {
        guard let indexPtr = indexPtr else {
            throw HNSWError.initializationFailed
        }
        
        let queryCount = query.count
        guard queryCount > 0 else {
            return ([], [])
        }
        
        let vecDim = query[0].count
        guard vecDim == dim else {
            throw HNSWError.invalidDimension
        }
        
        // Flatten the 2D array
        var flattenedQuery = [Float](repeating: 0, count: queryCount * dim)
        for i in 0..<queryCount {
            for j in 0..<dim {
                flattenedQuery[i * dim + j] = query[i][j]
            }
        }
        
        // Allocate arrays for results
        var resultLabels = [UInt64](repeating: 0, count: queryCount * k)
        var resultDistances = [Float](repeating: 0, count: queryCount * k)
        
        if !hnswlib_bf_index_search_knn(indexPtr, flattenedQuery, size_t(k), &resultLabels, &resultDistances, size_t(queryCount), Int32(numThreads)) {
            throw HNSWError.searchFailed
        }
        
        // Reshape results
        var labels = [[UInt64]](repeating: [UInt64](repeating: 0, count: k), count: queryCount)
        var distances = [[Float]](repeating: [Float](repeating: 0, count: k), count: queryCount)
        
        for i in 0..<queryCount {
            for j in 0..<k {
                labels[i][j] = resultLabels[i * k + j]
                distances[i][j] = resultDistances[i * k + j]
            }
        }
        
        return (labels, distances)
    }
}

// MARK: - Private C Interface

// These are the C wrapper functions from HNSWLibWrapper.cpp
// They are declared private to hide them from Swift users

@_silgen_name("hnswlib_index_create")
private func hnswlib_index_create(_ space_type: Int32, _ dim: Int32) -> OpaquePointer?

@_silgen_name("hnswlib_index_free")
private func hnswlib_index_free(_ index: OpaquePointer)

@_silgen_name("hnswlib_index_init")
private func hnswlib_index_init(_ index: OpaquePointer, _ max_elements: size_t, _ M: size_t, _ ef_construction: size_t, _ random_seed: size_t, _ allow_replace_deleted: Bool) -> Bool

@_silgen_name("hnswlib_index_add_items")
private func hnswlib_index_add_items(_ index: OpaquePointer, _ data: [Float], _ rows: size_t, _ dim: size_t, _ ids: [UInt64]? = nil, _ num_threads: Int32, _ replace_deleted: Bool) -> Bool

@_silgen_name("hnswlib_index_search_knn")
private func hnswlib_index_search_knn(_ index: OpaquePointer, _ query: [Float], _ k: size_t, _ result_labels: UnsafeMutablePointer<UInt64>, _ result_distances: UnsafeMutablePointer<Float>, _ query_count: size_t, _ num_threads: Int32) -> Bool

@_silgen_name("hnswlib_index_set_ef")
private func hnswlib_index_set_ef(_ index: OpaquePointer, _ ef: size_t)

@_silgen_name("hnswlib_index_get_current_count")
private func hnswlib_index_get_current_count(_ index: OpaquePointer) -> size_t

@_silgen_name("hnswlib_index_get_max_elements")
private func hnswlib_index_get_max_elements(_ index: OpaquePointer) -> size_t

@_silgen_name("hnswlib_index_get_ef")
private func hnswlib_index_get_ef(_ index: OpaquePointer) -> size_t

@_silgen_name("hnswlib_index_get_m")
private func hnswlib_index_get_m(_ index: OpaquePointer) -> size_t

@_silgen_name("hnswlib_index_save")
private func hnswlib_index_save(_ index: OpaquePointer, _ path: UnsafePointer<Int8>) -> Bool

@_silgen_name("hnswlib_index_load")
private func hnswlib_index_load(_ space_type: Int32, _ dim: Int32, _ path: UnsafePointer<Int8>, _ max_elements: size_t, _ allow_replace_deleted: Bool) -> OpaquePointer?

@_silgen_name("hnswlib_index_mark_deleted")
private func hnswlib_index_mark_deleted(_ index: OpaquePointer, _ label: UInt64)

@_silgen_name("hnswlib_index_unmark_deleted")
private func hnswlib_index_unmark_deleted(_ index: OpaquePointer, _ label: UInt64)

@_silgen_name("hnswlib_index_resize")
private func hnswlib_index_resize(_ index: OpaquePointer, _ new_size: size_t) -> Bool

@_silgen_name("hnswlib_bf_index_create")
private func hnswlib_bf_index_create(_ space_type: Int32, _ dim: Int32) -> OpaquePointer?

@_silgen_name("hnswlib_bf_index_free")
private func hnswlib_bf_index_free(_ index: OpaquePointer)

@_silgen_name("hnswlib_bf_index_init")
private func hnswlib_bf_index_init(_ index: OpaquePointer, _ max_elements: size_t) -> Bool

@_silgen_name("hnswlib_bf_index_add_items")
private func hnswlib_bf_index_add_items(_ index: OpaquePointer, _ data: [Float], _ rows: size_t, _ dim: size_t, _ ids: [UInt64]? = nil) -> Bool

@_silgen_name("hnswlib_bf_index_search_knn") 
private func hnswlib_bf_index_search_knn(_ index: OpaquePointer, _ query: [Float], _ k: size_t, _ result_labels: UnsafeMutablePointer<UInt64>, _ result_distances: UnsafeMutablePointer<Float>, _ query_count: size_t, _ num_threads: Int32) -> Bool
