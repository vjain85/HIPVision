#include "cuda_grayscale_grid_stride.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                        \
    do                                                                                          \
    {                                                                                           \
        cudaError_t error = (call);                                                             \
        if (error != cudaSuccess)                                                               \
        {                                                                                       \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));  \
        }                                                                                       \
    } while (0);                                                                                \

// Grid stride grayscale kernel
__global__ void grayscaleGridStrideKernel(const uint8_t* input, uint8_t* output, size_t totalPixels)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (; idx < totalPixels; idx += stride)
    {
        const uint8_t r = input[3 * idx];
        const uint8_t g = input[3 * idx + 1];
        const uint8_t b = input[3 * idx + 2];

        output[idx] = static_cast<uint8_t>((299u * r + 587u * g + 114u * b) / 1000u);
    }
}

__global__ void grayscaleKernelGridStride(const uint8_t* input, uint8_t* output, size_t totalPixels)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (; idx < totalPixels; idx += stride)
    {
        int r = input[3 * idx];
        int g = input[3 * idx + 1];
        int b = input[3 * idx + 2];

        output[idx] = static_cast<uint8_t>((299u * r + 587u * g + 114u * b) / 1000u);
    }
}

// Constructor
CudaGrayscaleGridStride::CudaGrayscaleGridStride(size_t pixels)
    : maxPixels(pixels)
{
    if (!maxPixels)
        throw std::invalid_argument("Max pixels must be greater than 0.");

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), maxPixels * 3 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), maxPixels * sizeof(uint8_t)));
}

CudaGrayscaleGridStride::~CudaGrayscaleGridStride()
{
    if (d_input)
        cudaFree(d_input);

    if (d_output)
        cudaFree(d_output);
}

Image CudaGrayscaleGridStride::process(const Image& input)
{
    if (input.channels != 3)
        throw std::invalid_argument("Input must have 3 channels!");

    const size_t totalPixels = input.pixelCount();

    if (totalPixels > maxPixels)
        throw std::invalid_argument("Input exceeds allocated capacity!");

    Image output(input.width, input.height, 1);

    const size_t inputBytes = totalPixels * 3 * sizeof(uint8_t);

    const size_t outputBytes = totalPixels * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpy(d_input, input.pixels(), inputBytes, cudaMemcpyHostToDevice));

    constexpr int threadPerBlock = 256;

    // Limit the grid for 256 blocks for this experiment
    // Grid stride loop handles all remaining pixels.
    constexpr int blocksPerGrid = 256;

    grayscaleGridStrideKernel<<< blocksPerGrid, threadPerBlock>>>(d_input, d_output, totalPixels);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.pixels(), d_output, outputBytes, cudaMemcpyDeviceToHost));

    return output;
}