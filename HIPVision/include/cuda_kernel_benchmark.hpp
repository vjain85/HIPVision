#pragma once

#include "image.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

class CudaKernelBenchmark
{
public:
    explicit CudaKernelBenchmark(size_t maxPixels);
    ~CudaKernelBenchmark();

    CudaKernelBenchmark(const CudaKernelBenchmark&) = delete;
    CudaKernelBenchmark& operator=(const CudaKernelBenchmark&) = delete;

    void upload(const Image& input);

    float benchmarkOnePixel(int iterations);
    float benchmarkGridStride(int blocks, int iterations);

    bool validate();

private:
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    size_t maxPixels = 0;
    size_t currentPixels = 0;

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
};