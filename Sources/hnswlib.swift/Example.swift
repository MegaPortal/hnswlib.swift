/* 
 * Example.swift - Example usage of HNSW Swift bindings
 *
 * This file provides examples of how to use the hnswlib.swift bindings.
 * This is meant to be used as a reference, not included in the library itself.
 * 
 * To use the hnswlib.swift package in your own project:
 *
 * 1. Add it as a dependency in your Package.swift:
 *    .package(url: "https://github.com/MegaPortal/hnswlib.swift.git", from: "1.0.0")
 *
 * 2. Import it in your code:
 *    import hnswlib_swift
 * 
 * 3. Use it as shown in the example below:
 */

/*
import Foundation
import hnswlib_swift  // Import the package

func hnswlibExample() {
    do {
        // Create a new index with L2 (Euclidean) distance
        let dimensions = 128
        let index = try HNSWIndex(spaceType: .l2, dim: dimensions)
        
        // Initialize the index
        let maxElements = 10000
        try index.initIndex(
            maxElements: maxElements,
            m: 16,                 // Number of bidirectional links
            efConstruction: 200,   // Size of dynamic list during construction
            randomSeed: 100,
            allowReplaceDeleted: false
        )
        
        // Set ef (higher values = more accurate search, but slower)
        index.setEf(ef: 50)
        
        // Create some random vectors for demonstration
        print("Generating random vectors...")
        var vectors: [[Float]] = []
        for i in 0..<1000 {
            var vector = [Float](repeating: 0, count: dimensions)
            for j in 0..<dimensions {
                vector[j] = Float.random(in: 0...1)
            }
            vectors.append(vector)
        }
        
        // Add vectors to the index
        print("Adding vectors to the index...")
        try index.addItems(data: vectors)
        
        // Check the current count
        print("Elements in index: \(index.currentCount)")
        
        // Query the index
        print("Searching for nearest neighbors...")
        let queryVector = vectors[0] // Use the first vector as a query
        let k = 5 // Return 5 nearest neighbors
        
        let startTime = Date()
        let results = try index.searchKnn(query: [queryVector], k: k)
        let elapsedTime = Date().timeIntervalSince(startTime)
        
        // Print results
        print("Query results (took \(elapsedTime) seconds):")
        for i in 0..<k {
            print("Neighbor \(i): ID = \(results.labels[0][i]), Distance = \(results.distances[0][i])")
        }
        
        // (Optional) Save the index
        let savePath = NSTemporaryDirectory() + "test_index.bin"
        print("Saving index to \(savePath)")
        try index.saveIndex(path: savePath)
        
        // (Optional) Load the index
        print("Loading index from \(savePath)")
        let loadedIndex = try HNSWIndex.loadIndex(
            spaceType: .l2,
            dim: dimensions,
            path: savePath
        )
        
        print("Loaded index has \(loadedIndex.currentCount) elements")
        
    } catch {
        print("Error: \(error)")
    }
}

// Call the example function
hnswlibExample()
*/ 