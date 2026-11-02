# HIPVision

A high-performance image processing engine
using C++ and AMD HIP.

## Current Features

- RGB image representation
- CPU grayscale conversion
- Synthetic image generation
- Execution time measurement

## Technologies

- C++17
- CMake
- AMD HIP (planned)
- CUDA

## Build

Configure and build Debug:

```bash
cmake -S . -B build/Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug
```

Configure and build Release alongside Debug:

```bash
cmake -S . -B build/Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release
```
