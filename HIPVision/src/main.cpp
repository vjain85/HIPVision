#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iomanip>
#include "backend.hpp"
#include "cuda_grayscale.hpp"

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

    std::cout << "HIPVision - Day 5\n";

    Image input(WIDTH, HEIGHT, 3);

    //Generate synthetic RGB image.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for(auto& pixel : input.data)
        pixel = static_cast<uint8_t>(dist(rng));

    std::cout << "Image: " << WIDTH << "x" << HEIGHT << std::endl;
        
    constexpr int iterations = 10;
        
    // CPU benchmark
    Image cpuOutput(WIDTH, HEIGHT, 1);
    std::cout << "\nCPU Benchmark (" << iterations << " iterations)\n";

    double cpuAverage = benchmark([&]() { 
        cpuOutput = grayscaleCPU(input);
    }, iterations);

    
    // CUDA benchmark
    CudaGrayscale cudaProcessor(input.pixelCount());

    Image cudaOutput(WIDTH, HEIGHT, 1);
    std::cout << "\nCUDA reusable memory Benchmark (" << iterations << " iterations)\n";

    double cudaAverage = benchmark([&]() { 
        cudaOutput = cudaProcessor.process(input);
    }, iterations);

    std::cout << "\nCUDA Transfer breakdown:\n";

    std::cout << "Host to Device time: " << cudaProcessor.getH2DTimeMs() << " ms\n";
    std::cout << "Kernel execution time: "<< cudaProcessor.getKernelTimeMs() << " ms\n";
    std::cout << "Device to Host time: "<< cudaProcessor.getD2HTimeMs() << " ms\n";

    std::cout << "Sum of GPU stages: " << (cudaProcessor.getH2DTimeMs() + cudaProcessor.getKernelTimeMs() + cudaProcessor.getD2HTimeMs()) << " ms\n";

    std::cout << "\nCorrectness: "<< (cpuOutput.data == cudaOutput.data ? "PASS" : "FAIL") << '\n';

    std::cout << "\nAverage CPU time: " << cpuAverage << " ms\n";
    std::cout << "Average CUDA end-to-end time: "<< cudaAverage << " ms\n";

    std::cout << "End-to-end speedup: "<< cpuAverage / cudaAverage << "x\n";

    return 0;
}