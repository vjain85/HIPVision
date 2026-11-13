#pragma once

#include <pinned_image.hpp>
#include <cuda_runtime.h>

#include <stddef.h>
#include <cstdint>

class CudaGrayscaleMultiStream
{
public:
    CudaGrayscaleMultiStream(size_t maxPixels, size_t chunkPixels = 2);
    ~CudaGrayscaleMultiStream();

    CudaGrayscaleMultiStream(const CudaGrayscaleMultiStream&) = delete;
    CudaGrayscaleMultiStream& operator=(const CudaGrayscaleMultiStream&) = delete;

    void process(const PinnedImage& input, PinnedImage& output);
    float getTotalTimeMs() const { return totalTimeMs; }

private:
    size_t maxPixels =0;
    size_t chunkPixels = 0;

    static constexpr int STREAM_COUNT = 2;

    uint8_t* d_input[STREAM_COUNT] = {nullptr, nullptr};
    uint8_t* d_output[STREAM_COUNT] = {nullptr, nullptr};

    cudaStream_t streams[STREAM_COUNT] = {nullptr, nullptr};

    float totalTimeMs = 0.0f;

};
