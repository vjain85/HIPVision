#pragma once

#include "image.hpp"

#include <cuda_runtime.h>
#include <cstddef>

class CudaGrayscalePinned
{
public:
    CudaGrayscalePinned(size_t maxPixels);
    ~CudaGrayscalePinned();

    CudaGrayscalePinned(const CudaGrayscalePinned&) = delete;
    CudaGrayscalePinned& operator=(const CudaGrayscalePinned&) = delete;

    Image process(const Image& input);

    float getH2DTimeMs() const { return h2dTimeMs; }
    float getD2HTimeMs() const { return d2hTimeMs; }
    float getKernelTimeMs() const { return kernelTimeMs; }

    float getInputCopyTimeMs() const { return inputCopyTimeMs; }
    float getOutputCopyTimeMs() const { return outputCopyTimeMs; }

private:
    // GPU buffers
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    // Pinned host buffers
    uint8_t* h_input = nullptr;
    uint8_t* h_output = nullptr;

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

    float inputCopyTimeMs = 0.0f;
    float outputCopyTimeMs = 0.0f;
};