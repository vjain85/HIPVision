#include "cuda_grayscale_multistream.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err = (call);                                                               \                
        if (err != cudaSuccess)                                                                 \
        {                                                                                       \        
            throw std::runtime_error(std::string("CUDA ERROR: ") + cudaGetErrorString(err));    \
        }                                                                                       \
    }while(0)                                                                                   \

__global__ void grayScaleMultiStreamKernel(const uint8_t* input, uint8_t* output, size_t pixelCount)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelCount)
    {
        const uint8_t r = input[3 * idx];
        const uint8_t g = input[3 * idx + 1];
        const uint8_t b = input[3 * idx + 2];

        output[idx] = static_cast<uint8_t>((299u * r + 587u * g + 114u * b) / 1000u);
    }
}

CudaGrayscaleMultiStream::CudaGrayscaleMultiStream(size_t pixels, size_t chunkSize)
    : maxPixels(pixels), chunkPixels(chunkSize)
{
    if (maxPixels == 0 || chunkPixels == 0)
        throw std::invalid_argument("Invalid image or chunk size.");

    try
    {
        for (int i = 0; i < STREAM_COUNT; ++i)
        {
            // Allocate device input buffer for this stream
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input[i]), 3 * chunkPixels * sizeof(uint8_t)));

            // Allocate device output buffer for this stream
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output[i]), chunkPixels * sizeof(uint8_t)));

            // Create independent cuda stream
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
    }
    catch(...)
    {
       for (int i = 0; i < STREAM_COUNT; ++i)
       {
            if (streams[i])
                cudaStreamDestroy(streams[i]);

            if (d_input[i])
               cudaFree(d_input[i]);

            if (d_output[i])
                cudaFree(d_output[i]);
       }
       throw;
    }
}

// Desructor
CudaGrayscaleMultiStream::~CudaGrayscaleMultiStream()
{
    for (int i = 0; i < STREAM_COUNT; ++i)
        if (streams[i])
            cudaStreamSynchronize(streams[i]);

    for (int i = 0; i < STREAM_COUNT; ++i)
    {
        if (streams[i])
            cudaStreamDestroy(streams[i]);

        if (d_input[i])
            cudaFree(d_input[i]);

        if (d_output[i])
            cudaFree(d_output[i]);
    }
}

void CudaGrayscaleMultiStream::process(const PinnedImage& input, PinnedImage& output)
{
    if (input.channels != 3)
        throw std::invalid_argument("Input must have 3 channels.");

    if (output.channels != 1 || input.height != output.height || input.width != output.width)
        throw std::invalid_argument("Invalid output image.");

    const size_t totalPixels = input.pixelCount();

    if (totalPixels > maxPixels)
        throw std::invalid_argument("Input exceeds allocated capacity.");

    constexpr int threadsPerBlock = 256;

    for (size_t offset = 0, chunkIndex = 0; offset < totalPixels; offset += chunkPixels, ++chunkIndex)
    {
        const int streamIndex = chunkIndex % STREAM_COUNT;

        const size_t currentChunkPixels = std::min(chunkPixels, totalPixels - offset);

        const size_t inputBytes = currentChunkPixels * 3 * sizeof(uint8_t);

        const size_t outputBytes = currentChunkPixels * sizeof(uint8_t);

        const uint8_t* hostInput = input.pixels() + offset * 3;
        uint8_t* hostOutput = output.pixels() + offset;

        CUDA_CHECK(cudaMemcpyAsync(d_input[streamIndex], hostInput, inputBytes, cudaMemcpyHostToDevice, streams[streamIndex]));

        const int blocksPerGrid = static_cast<int>((currentChunkPixels + threadsPerBlock - 1)/threadsPerBlock);

        grayScaleMultiStreamKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input[streamIndex], d_output[streamIndex], currentChunkPixels);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(hostOutput, d_output[streamIndex], outputBytes, cudaMemcpyDeviceToHost, streams[streamIndex]));
    }

    for (int i = 0; i < STREAM_COUNT; ++i)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
}