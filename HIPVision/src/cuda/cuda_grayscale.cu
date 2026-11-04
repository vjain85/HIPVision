#include "cuda_grayscale.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            throw std::runtime_error(                         \
                std::string("CUDA error: ") +                 \
                cudaGetErrorString(err));                     \
        }                                                     \
    } while (0)

CudaGrayscale::CudaGrayscale(size_t pixels)
    : maxPixels(pixels)
{
    try
    {
    CUDA_CHECK(cudaMalloc((void**)&d_input, maxPixels * 3 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, maxPixels * sizeof(uint8_t)));
    }
    catch (const std::exception& e)
    {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        throw std::runtime_error(std::string("Failed to allocate device memory: ") + e.what());
    }
    catch (...)
    {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        throw std::runtime_error("Failed to allocate device memory due to an unknown error.");
    }
}

CudaGrayscale::~CudaGrayscale()
{
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
}

__global__ void grayscaleKernelReusable(const uint8_t* input, uint8_t* output, size_t pixelCount)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);

    if (idx < pixelCount) {
        int r = input[idx * 3];
        int g = input[idx * 3 + 1];
        int b = input[idx * 3 + 2];

        output[idx] = static_cast<uint8_t>((299u * r + 587u * g + 114u * b) / 1000u);
    }
}

Image CudaGrayscale::process(const Image& input)
{
    if (input.channels != 3) {
        throw std::runtime_error("Input image must have 3 channels.");
    }

    const size_t pixelCount = input.pixelCount();

    if (pixelCount > maxPixels) {
        throw std::runtime_error("Image exceeds allocated CUDA buffer.");
    }

    Image output(input.width, input.height, 1);

    size_t inputSize = pixelCount * 3 * sizeof(uint8_t);
    size_t outputSize = pixelCount * sizeof(uint8_t);

    try
    {
        CUDA_CHECK(cudaMemcpy(d_input, input.pixels(), inputSize, cudaMemcpyHostToDevice));

        constexpr int threadsPerBlock = 256;
        const int blocksPerGrid = static_cast<int>((pixelCount + threadsPerBlock - 1) / threadsPerBlock);

        grayscaleKernelReusable<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, pixelCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.pixels(), d_output, outputSize, cudaMemcpyDeviceToHost));
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("CUDA processing failed: ") + e.what());
    }

    return output;
}