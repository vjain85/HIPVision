#include "cuda_grayscale_pinned.hpp"

#include <cstring>
#include <string>
#include <stdexcept>
#include <chrono>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            throw std::runtime_error(                         \
                std::string("CUDA error: ") +                 \
                cudaGetErrorString(err));                     \
        }                                                     \
    } while (0)

__global__ void grayscalePinnedKernel(const uint8_t* input, uint8_t* output, size_t pixelCount)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx < pixelCount)
    {
        uint8_t r = input[3 * idx];
        uint8_t g = input[3 * idx + 1];
        uint8_t b = input[3 * idx + 2];
        output[idx] = static_cast<uint8_t>((299u * r + 587u * g + 114u * b) / 1000u);
    }
}

CudaGrayscalePinned::CudaGrayscalePinned(size_t pixels)
    : maxPixels(pixels)
{
    try
    {
        // Allocate pinned host memory
        CUDA_CHECK(cudaMallocHost(&h_input, maxPixels * 3 * sizeof(uint8_t)));
        CUDA_CHECK(cudaMallocHost(&h_output, maxPixels * sizeof(uint8_t)));

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_input, maxPixels * 3 * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_output, maxPixels * sizeof(uint8_t)));

        // Create a stream for asynchronous operations
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Create events for timing
        CUDA_CHECK(cudaEventCreate(&h2dStart));
        CUDA_CHECK(cudaEventCreate(&h2dStop));
        
        CUDA_CHECK(cudaEventCreate(&d2hStart));
        CUDA_CHECK(cudaEventCreate(&d2hStop));

        CUDA_CHECK(cudaEventCreate(&kernelStart));
        CUDA_CHECK(cudaEventCreate(&kernelStop));
    }
    catch (const std::exception& e)
    {
        // Constructor failed, clean up any allocated resources
        if (h_input) cudaFreeHost(h_input);
        if (h_output) cudaFreeHost(h_output);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
        if (h2dStart) cudaEventDestroy(h2dStart);
        if (h2dStop) cudaEventDestroy(h2dStop);
        if (d2hStart) cudaEventDestroy(d2hStart);
        if (d2hStop) cudaEventDestroy(d2hStop);
        if (kernelStart) cudaEventDestroy(kernelStart);
        if (kernelStop) cudaEventDestroy(kernelStop);
        throw; // Rethrow the exception to indicate failure
    }
}

// Destructor
CudaGrayscalePinned::~CudaGrayscalePinned()
{
    if (h_input) cudaFreeHost(h_input);
    if (h_output) cudaFreeHost(h_output);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (stream) cudaStreamDestroy(stream);
    if (h2dStart) cudaEventDestroy(h2dStart);
    if (h2dStop) cudaEventDestroy(h2dStop);
    if (d2hStart) cudaEventDestroy(d2hStart);
    if (d2hStop) cudaEventDestroy(d2hStop);
    if (kernelStart) cudaEventDestroy(kernelStart);
    if (kernelStop) cudaEventDestroy(kernelStop);
}

Image CudaGrayscalePinned::process(const Image& input)
{
    if (input.channels != 3)
    {
        throw std::invalid_argument("Input image must have 3 channels (RGB).");
    }

    const size_t pixelCount = input.width * input.height;

    if (pixelCount > maxPixels)
        throw std::runtime_error("Input image exceeds allocated CUDA buffer size.");
    
    const size_t inputSize = pixelCount * 3 * sizeof(uint8_t);
    const size_t outputSize = pixelCount * sizeof(uint8_t);

    Image output(input.width, input.height, 1);

    try
    {
        auto copyStart = std::chrono::steady_clock::now();
        // Copy pageable Image data to pinned host memory
        std::memcpy(h_input, input.pixels(), inputSize);
        auto copyEnd = std::chrono::steady_clock::now();
        inputCopyTimeMs = std::chrono::duration<float, std::milli>(copyEnd - copyStart).count();

        // Record the start event for H2D transfer
        CUDA_CHECK(cudaEventRecord(h2dStart, stream));

        // Asynchronously copy data from host to device
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, inputSize, cudaMemcpyHostToDevice, stream));

        // Record the stop event for H2D transfer
        CUDA_CHECK(cudaEventRecord(h2dStop, stream));

        // Launch the kernel
        constexpr int threadsPerBlock = 256;
        const int blocksPerGrid = static_cast<int>((pixelCount + threadsPerBlock - 1) / threadsPerBlock);

        // Record the start event for kernel execution
        CUDA_CHECK(cudaEventRecord(kernelStart, stream));

        grayscalePinnedKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, pixelCount);
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

        // Record the stop event for kernel execution
        CUDA_CHECK(cudaEventRecord(kernelStop, stream));

        // Record the start event for D2H transfer
        CUDA_CHECK(cudaEventRecord(d2hStart, stream));

        // Asynchronously copy data from device to host
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, outputSize, cudaMemcpyDeviceToHost, stream));

        // Record the stop event for D2H transfer
        CUDA_CHECK(cudaEventRecord(d2hStop, stream));

        // Wait for all operations in the stream to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventElapsedTime(&h2dTimeMs, h2dStart, h2dStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&d2hTimeMs, d2hStart, d2hStop));
        
        // Copy the result from pinned host memory to the output Image
        copyStart = std::chrono::steady_clock::now();
        std::memcpy(output.pixels(), h_output, outputSize);
        copyEnd = std::chrono::steady_clock::now();
        outputCopyTimeMs = std::chrono::duration<float, std::milli>(copyEnd - copyStart).count();

    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("CUDA processing failed: ") + e.what());
    }
    return output;
}