#pragma once

#include "pinned_image.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

class CudaGrayscaleDirect
{
public:
    CudaGrayscaleDirect(size_t maxPixels);
    ~CudaGrayscaleDirect();

    CudaGrayscaleDirect(const CudaGrayscaleDirect&) = delete;
    CudaGrayscaleDirect& operator=(const CudaGrayscaleDirect&) = delete;

    PinnedImage process(const PinnedImage& input);

    float getH2DTimeMs() const { return h2dTimeMs; }
    float getD2HTimeMs() const { return d2hTimeMs; }
    float getKernelTimeMs() const { return kernelTimeMs; }

    private:
    // GPU buffers
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    size_t maxPixels = 0;

    cudaStream_t stream = nullptr;

    // Events for timing
    cudaEvent_t h2dStart = nullptr;
    cudaEvent_t h2dStop = nullptr;
    cudaEvent_t d2hStart = nullptr;
    cudaEvent_t d2hStop = nullptr;
    cudaEvent_t kernelStart = nullptr;
    cudaEvent_t kernelStop = nullptr;

    float h2dTimeMs = 0.0f;
    float d2hTimeMs = 0.0f;
    float kernelTimeMs = 0.0f;
};