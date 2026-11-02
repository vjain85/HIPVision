#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

__global__ void rgbToGrayscaleKernel(const uint8_t* input, uint8_t* output, int pixelCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelCount) {
        int r = input[idx * 3];
        int g = input[idx * 3 + 1];
        int b = input[idx * 3 + 2];
        output[idx] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main()
{
    const int pixelCount = 4;
    const int inputSize = pixelCount * 3; // RGB

    uint8_t h_input[inputSize] = { 255, 0, 0,   // Red
                                0, 255, 0,   // Green
                                0, 0, 255,   // Blue
                                255, 255, 255 }; // White

    uint8_t h_output[pixelCount] = {}; // Grayscale output

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    cudaError_t err = cudaMalloc((void**)&d_input, inputSize * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return -1;
    }

    err = cudaMalloc((void**)&d_output, pixelCount * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    err = cudaMemcpy(d_input, h_input, inputSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }   

    int threadsPerBlock = 256;
    int blocksPerGrid = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;

    rgbToGrayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, pixelCount);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {       
        std::cerr << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    err = cudaMemcpy(h_output, d_output, pixelCount * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    std::cout << "Grayscale values: ";
    for (int i = 0; i < pixelCount; ++i) {
        std::cout << static_cast<int>(h_output[i]) << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}