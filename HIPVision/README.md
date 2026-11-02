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

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build