// swift-tools-version: 5.10.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "hnswlib.swift",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "hnswlib_swift",
            targets: ["hnswlib_swift"]),
    ],
    targets: [
        // C++ wrapper target
        .target(
            name: "hnswlib_cpp",
            path: "Sources/CppSources",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("../hnswlib.cpp"),
                .define("NDEBUG"),
                .unsafeFlags(["-std=c++11"])
            ]
        ),
        // Swift target
        .target(
            name: "hnswlib_swift",
            dependencies: ["hnswlib_cpp"],
            path: "Sources/hnswlib.swift"
        ),
        .testTarget(
            name: "hnswlib.swiftTests",
            dependencies: ["hnswlib_swift"]
        ),
    ]
)
