#include "cuda_kernel_benchmark.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                        \
    do                                                                                          \
    {                                                                                           \
        cudaError_t error = (call);                                                             \
        if (error != cudaSuccess)                                                               \
        {                                                                                       \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));  \
        }                                                                                       \
    } while (0);                                                                                \

__global__ void grayscaleOnePixelKernel(const uint8_t* input, uint8_t* output, size_t pixelCount)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < pixelCount)
    {
        const int r = input[idx * 3];
        const int g = input[idx * 3 + 1];
        const int b = input[idx * 3 + 2];

        output[idx] = static_cast<uint8_t>((299u * r + 587u * g + 114u * b) / 1000u);
    }
}

__global__ void grayscaleKernelGridStrideBenchmark(const uint8_t* input, uint8_t* output, size_t totalPixels)
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
CudaKernelBenchmark::CudaKernelBenchmark(size_t maxPixels)
    : maxPixels(maxPixels)
{
    if (maxPixels == 0)
        throw std::invalid_argument("max pixels must be greater than 0.");

    try
    {
        CUDA_CHECK(cudaMalloc((&d_input), maxPixels * 3 * sizeof(uint8_t)));

        CUDA_CHECK(cudaMalloc((&d_output), maxPixels * sizeof(uint8_t)));

        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }
    catch(...)
    {
        if (startEvent)
            cudaEventDestroy(startEvent);
        if (stopEvent)
            cudaEventDestroy(stopEvent);
        if (d_input)
            cudaFree(d_input);
        if(d_output)
            cudaFree(d_output);

        throw;
    }
}

//Destructor
CudaKernelBenchmark::~CudaKernelBenchmark()
{
    if (startEvent)
        cudaEventDestroy(startEvent);
    if (stopEvent)
        cudaEventDestroy(stopEvent);
    if (d_input)
        cudaFree(d_input);
    if(d_output)
        cudaFree(d_output);
}

void CudaKernelBenchmark::upload(const Image& input)
{
    if (input.channels != 3)
        throw std::invalid_argument("Input must have 3 channels.");

    const size_t pixels = input.pixelCount();

    if (pixels > maxPixels)
        throw std::invalid_argument("Input exceeds allocated capacity.");

    const size_t inputBytes = pixels * 3 * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpy(d_input, input.pixels(), inputBytes, cudaMemcpyHostToDevice));

    currentPixels = pixels;
}

float CudaKernelBenchmark::benchmarkOnePixel(int iterations)
{
    if (currentPixels == 0)
        throw std::runtime_error("Upload an image before benchmarking!");

    if (iterations <= 0)
        throw std::invalid_argument("Iterations must be greater than zero!");

    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks = static_cast<int>((currentPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    //Warm-up launch
    grayscaleOnePixelKernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, currentPixels);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int i = 0; i < iterations; ++i)
    {
        grayscaleOnePixelKernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, currentPixels);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float totalTimeMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&totalTimeMs, startEvent, stopEvent));

    return totalTimeMs / iterations;
}

float CudaKernelBenchmark::benchmarkGridStride(int blocks, int iterations)
{
    if (currentPixels == 0)
        throw std::runtime_error("Üpload an image before benchmarking.");

    if (blocks <= 0 || iterations <= 0)
        throw std::invalid_argument("blocks and iterations must be positive.");

    constexpr int THREADS_PER_BLOCK = 256;

    //warm up
    grayscaleKernelGridStrideBenchmark<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, currentPixels);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int i = 0; i < iterations; ++i)
    {
        grayscaleKernelGridStrideBenchmark<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, currentPixels);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float totalTimeMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&totalTimeMs, startEvent, stopEvent));

    return totalTimeMs / iterations;
}

bool CudaKernelBenchmark::validate()
{
    if (currentPixels == 0)
        throw std::runtime_error("Upload an image before validation.");

    std::vector<uint8_t> output(currentPixels);

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, currentPixels * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    return !output.empty();
}