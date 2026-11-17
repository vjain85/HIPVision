#pragma once

#include "image.hpp"
#include <cstddef>
#include <cstdint>

class CudaGrayscaleGridStride
{
public:
    explicit CudaGrayscaleGridStride(size_t maxPixels);
    ~CudaGrayscaleGridStride();

    CudaGrayscaleGridStride(const CudaGrayscaleGridStride&) = delete;
    CudaGrayscaleGridStride& operator=(const CudaGrayscaleGridStride&) = delete;

    Image process(const Image& input);

private:

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    size_t maxPixels = 0;
};