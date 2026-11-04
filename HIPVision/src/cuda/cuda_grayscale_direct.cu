#include "cuda_grayscale_direct.hpp"

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

__global__ void grayscaleDirectKernel(const uint8_t* input, uint8_t* output, int pixelCount)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < pixelCount)
    {
        uint8_t r = input[3 * idx];
        uint8_t g = input[3 * idx + 1];
        uint8_t b = input[3 * idx + 2];
        output[idx] = static_cast<uint8_t>((299u * r + 587u * g + 114u * b) / 1000u);
    }
}

CudaGrayscaleDirect::CudaGrayscaleDirect(size_t pixels)
    : maxPixels(pixels)
{
    try
    {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), 3 * maxPixels * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), maxPixels * sizeof(uint8_t)));

        // Create a CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Create events for timing
        CUDA_CHECK(cudaEventCreate(&h2dStart));
        CUDA_CHECK(cudaEventCreate(&h2dStop));

        CUDA_CHECK(cudaEventCreate(&d2hStart));
        CUDA_CHECK(cudaEventCreate(&d2hStop));
        
        CUDA_CHECK(cudaEventCreate(&kernelStart));
        CUDA_CHECK(cudaEventCreate(&kernelStop));
    }
    catch(...)
    {
        // Free any allocated resources in case of an exception
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
        if (h2dStart) cudaEventDestroy(h2dStart);
        if (h2dStop) cudaEventDestroy(h2dStop);
        if (d2hStart) cudaEventDestroy(d2hStart);
        if (d2hStop) cudaEventDestroy(d2hStop);
        if (kernelStart) cudaEventDestroy(kernelStart);
        if (kernelStop) cudaEventDestroy(kernelStop);

        throw; // rethrow the exception
    }
}

CudaGrayscaleDirect::~CudaGrayscaleDirect()
{
    if (stream) cudaStreamSynchronize(stream); // Ensure all operations are complete before freeing resources

    if (h2dStart) cudaEventDestroy(h2dStart);
    if (h2dStop) cudaEventDestroy(h2dStop);

    if (kernelStart) cudaEventDestroy(kernelStart);
    if (kernelStop) cudaEventDestroy(kernelStop);

    if (d2hStart) cudaEventDestroy(d2hStart);
    if (d2hStop) cudaEventDestroy(d2hStop);

    if (stream) cudaStreamDestroy(stream);

    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
}

PinnedImage CudaGrayscaleDirect::process(const PinnedImage& input)
{
    if (input.channels != 3)
    {
        throw std::invalid_argument("Input image must have 3 channels (RGB).");
    }

    const size_t inputPixelCount = input.pixelCount();
    if (inputPixelCount > maxPixels)
    {
        throw std::invalid_argument("Input image exceeds allocated CUDA buffer size.");
    }
    
    size_t inputSize = 3 * input.pixelCount() * sizeof(uint8_t);
    size_t outputSize = input.pixelCount() * sizeof(uint8_t);
    
    PinnedImage output(input.width, input.height, 1);

    // Record the start event for H2D transfer
    CUDA_CHECK(cudaEventRecord(h2dStart, stream));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, input.pixels(), inputSize, cudaMemcpyHostToDevice, stream));

    // Record the stop event for H2D transfer
    CUDA_CHECK(cudaEventRecord(h2dStop, stream));

    // Launch the kernel
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = static_cast<int>((input.pixelCount() + threadsPerBlock - 1) / threadsPerBlock);

    // Record the start event for kernel execution
    CUDA_CHECK(cudaEventRecord(kernelStart, stream));

    grayscaleDirectKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, static_cast<int>(input.pixelCount()));

    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    // Record the stop event for kernel execution
    CUDA_CHECK(cudaEventRecord(kernelStop, stream));

    // Record the start event for D2H transfer
    CUDA_CHECK(cudaEventRecord(d2hStart, stream));

    // Copy output data back to host
    CUDA_CHECK(cudaMemcpyAsync(output.pixels(), d_output, outputSize, cudaMemcpyDeviceToHost, stream));

    // Record the stop event for D2H transfer
    CUDA_CHECK(cudaEventRecord(d2hStop, stream));

    // Wait for all operations in the stream to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Calculate elapsed times
    CUDA_CHECK(cudaEventElapsedTime(&h2dTimeMs, h2dStart, h2dStop));
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
    CUDA_CHECK(cudaEventElapsedTime(&d2hTimeMs, d2hStart, d2hStop));

    return output;
}