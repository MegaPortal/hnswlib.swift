# hnswlib.swift

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FMegaPortal%2Fhnswlib.swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/MegaPortal/hnswlib.swift)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FMegaPortal%2Fhnswlib.swift%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/MegaPortal/hnswlib.swift)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub tag](https://img.shields.io/github/tag/MegaPortal/hnswlib.swift.svg)](https://GitHub.com/MegaPortal/hnswlib.swift/tags/)

Swift bindings for the [HNSW (Hierarchical Navigable Small World)](https://github.com/nmslib/hnswlib) library, a fast approximate nearest neighbor search.

## Features

- High-performance nearest neighbor search in high-dimensional spaces
- Support for different distance metrics:
  - L2 (Euclidean distance)
  - Inner Product (cosine similarity on normalized vectors)
  - Cosine similarity (vectors are automatically normalized)
- Thread-safe, multi-threaded index construction and queries
- Save and load indices from disk
- Support for element deletion
- Swift-friendly API

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/MegaPortal/hnswlib.swift.git", from: "1.0.0")
]
```

Then add it to your target dependencies:

```swift
targets: [
    .target(
        name: "YourTarget",
        dependencies: ["hnswlib_swift"]
    )
]
```

## Usage

### Basic Example

```swift
import Foundation
import hnswlib_swift

// Create a new index with L2 (Euclidean) distance
let dimensions = 128
let index = try HNSWIndex(spaceType: .l2, dim: dimensions)

// Initialize the index
let maxElements = 10000
try index.initIndex(
    maxElements: maxElements,
    m: 16,                 // Number of bidirectional links
    efConstruction: 200,   // Size of dynamic list during construction
    randomSeed: 100
)

// Set ef parameter (controls search accuracy vs. speed)
index.setEf(ef: 50)

// Create some sample vectors
var vectors: [[Float]] = []
// ... populate vectors ...

// Add vectors to the index
try index.addItems(data: vectors)

// Query the index
let queryVector = vectors[0] // Use the first vector as a query
let k = 5 // Return 5 nearest neighbors
let results = try index.searchKnn(query: [queryVector], k: k)

// Process results
for i in 0..<k {
    print("Neighbor \(i): ID = \(results.labels[0][i]), Distance = \(results.distances[0][i])")
}
```

### Saving and Loading Indices

```swift
// Save the index to a file
try index.saveIndex(path: "/path/to/index.bin")

// Load the index from a file
let loadedIndex = try HNSWIndex.loadIndex(
    spaceType: .l2,
    dim: dimensions,
    path: "/path/to/index.bin"
)
```

### Deleting Elements

```swift
// Mark an item as deleted
index.markDeleted(label: 123)

// Unmark a previously deleted item
index.unmarkDeleted(label: 123)
```

### Using Cosine Similarity

```swift
// Create an index with cosine similarity
let index = try HNSWIndex(spaceType: .cosine, dim: dimensions)

// The vectors will be automatically normalized
try index.addItems(data: vectors)
```

## Parameters

- **dim**: The dimensionality of the vectors
- **m**: The number of bidirectional links created for each element during construction (default: 16)
- **efConstruction**: The size of the dynamic list for the nearest neighbors during index construction (default: 200)
- **ef**: The size of the dynamic list for the nearest neighbors during search (default: 10)

## Performance Considerations

- Larger `ef` and `efConstruction` values provide better recall at the cost of longer construction/search times
- The `m` parameter controls the trade-off between memory consumption and search performance
- For optimal performance with large datasets, adjust the number of threads based on your hardware

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This is a Swift wrapper around the [hnswlib](https://github.com/nmslib/hnswlib) C++ library by Yury Malkov and Dmitry Yashunin.