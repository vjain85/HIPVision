#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iomanip>
#include "backend.hpp"
#include "cuda_grayscale.hpp"
#include "cuda_grayscale_pinned.hpp"
#include "cuda_grayscale_direct.hpp"

using Clock = std::chrono::steady_clock;

Image grayscaleCPU(const Image& input);

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool compareImages(const Image& img1, const Image& img2)
{
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels)
        return false;

    return std::equal(img1.data.begin(), img1.data.end(), img2.data.begin());
}

bool compareImages(const PinnedImage& img1, const Image& img2)
{
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels)
        return false;

    return std::equal(img1.pixels(), img1.pixels() + img1.sizeInBytes(), img2.pixels());
}

template<typename Function>
double benchmark(Function function, int iterations = 10)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warm-up run
    function();

    for (int i = 0; i < iterations; ++i)
    {
        auto start = Clock::now();
        function();
        auto end = Clock::now();
        times.emplace_back(elapsedMs(start, end));
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double average = sum / static_cast<double>(iterations);

    auto minTime = *std::min_element(times.begin(), times.end());
    auto maxTime = *std::max_element(times.begin(), times.end());

    std::cout << "Min Time: " << minTime << " ms\n";
    std::cout << "Max Time: " << maxTime << " ms\n";
    std::cout << "Average Time: " << average << " ms\n";

    return average;
}

int main()
{
    constexpr int WIDTH = 4096;
    constexpr int HEIGHT = 4096;

    std::cout << "HIPVision - Day 7\n";

    Image input(WIDTH, HEIGHT, 3);

    //Generate synthetic RGB image.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for(auto& pixel : input.data)
        pixel = static_cast<uint8_t>(dist(rng));

    std::cout << "Image: " << WIDTH << "x" << HEIGHT << std::endl;
        
    constexpr int iterations = 10;
        
    // 1. CPU benchmark
    Image cpuOutput(WIDTH, HEIGHT, 1);
    std::cout << "\nCPU Benchmark (" << iterations << " iterations)\n";

    double cpuAverage = benchmark([&]() { 
        cpuOutput = grayscaleCPU(input);
    }, iterations);

    
    // 2. CUDA pageable memory benchmark
    CudaGrayscale cudaProcessor(input.pixelCount());

    Image cudaOutput(WIDTH, HEIGHT, 1);
    std::cout << "\nCUDA pageable memory Benchmark (" << iterations << " iterations)\n";

    double cudaAverage = benchmark([&]() { 
        cudaOutput = cudaProcessor.process(input);
    }, iterations);

    std::cout << "\nPageable CUDA Transfer breakdown:\n";

    std::cout << "Host to Device time: " << cudaProcessor.getH2DTimeMs() << " ms\n";
    std::cout << "Kernel execution time: "<< cudaProcessor.getKernelTimeMs() << " ms\n";
    std::cout << "Device to Host time: "<< cudaProcessor.getD2HTimeMs() << " ms\n";

    // 3. CUDA pinned memory benchmark
    CudaGrayscalePinned cudaPinnedProcessor(input.pixelCount());

    Image cudaPinnedOutput(WIDTH, HEIGHT, 1);
    std::cout << "\nCUDA pinned memory Benchmark (" << iterations << " iterations)\n";

    double cudaPinnedAverage = benchmark([&]() { 
        cudaPinnedOutput = cudaPinnedProcessor.process(input);
    }, iterations);
    
    std::cout << "\nPinned CUDA Transfer breakdown:\n";
    std::cout << "Host to Device time: " << cudaPinnedProcessor.getH2DTimeMs() << " ms\n";
    std::cout << "Kernel execution time: "<< cudaPinnedProcessor.getKernelTimeMs() << " ms\n";
    std::cout << "Device to Host time: "<< cudaPinnedProcessor.getD2HTimeMs() << " ms\n";

    std::cout << "Input copy time (host to pinned): " << cudaPinnedProcessor.getInputCopyTimeMs() << " ms\n";
    std::cout << "Output copy time (pinned to host): " << cudaPinnedProcessor.getOutputCopyTimeMs() << " ms\n";

    // 4. CUDA direct pinned memory benchmark
    CudaGrayscaleDirect cudaDirectProcessor(input.pixelCount());
    PinnedImage cudaDirectInput(WIDTH, HEIGHT, 3);
    PinnedImage cudaDirectOutput(WIDTH, HEIGHT, 1);

    std::copy(input.data.begin(), input.data.end(), cudaDirectInput.pixels());

    std::cout << "\nCUDA direct pinned memory Benchmark (" << iterations << " iterations)\n";

    double cudaDirectAverage = benchmark([&]() { 
        cudaDirectOutput = cudaDirectProcessor.process(cudaDirectInput);
    }, iterations);

    std::cout << "\nDirect Pinned CUDA Transfer breakdown:\n";
    std::cout << "Host to Device time: " << cudaDirectProcessor.getH2DTimeMs() << " ms\n";
    std::cout << "Kernel execution time: "<< cudaDirectProcessor.getKernelTimeMs() << " ms\n";
    std::cout << "Device to Host time: "<< cudaDirectProcessor.getD2HTimeMs() << " ms\n";

    // 4. correctness check
    std::cout << "\nCorrectness:\n";

    std::cout << "CPU vs CUDA pageable memory: " << (compareImages(cpuOutput, cudaOutput) ? "Match" : "Mismatch") << std::endl;
    std::cout << "CPU vs CUDA pinned memory: " << (compareImages(cpuOutput, cudaPinnedOutput) ? "Match" : "Mismatch") << std::endl;
    std::cout << "CPU vs CUDA direct pinned memory: " << (compareImages(cudaDirectOutput, cpuOutput) ? "Match" : "Mismatch") << std::endl;

    // 5. Performance comparison
    std::cout << "\n=========Performance summary==========:\n";

    std::cout << "Average CPU time: " << cpuAverage << " ms\n";
    std::cout << "Average CUDA pageable memory time: "<< cudaAverage << " ms\n";
    std::cout << "Average CUDA pinned memory time: "<< cudaPinnedAverage << " ms\n";
    std::cout << "Average CUDA direct pinned memory time: "<< cudaDirectAverage << " ms\n";

    std::cout << "Speedup (CPU vs CUDA pageable memory): "<< cpuAverage / cudaAverage << "x\n";
    std::cout << "Speedup (CPU vs CUDA pinned memory): "<< cpuAverage / cudaPinnedAverage << "x\n";
    std::cout << "Speedup (CUDA pageable memory vs CUDA pinned memory): "<< cudaAverage / cudaPinnedAverage << "x\n";
    std::cout << "Speedup (CPU vs CUDA direct pinned memory): "<< cpuAverage / cudaDirectAverage << "x\n";

    return 0;
}