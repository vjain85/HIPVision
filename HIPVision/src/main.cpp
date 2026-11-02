#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>

#include "backend.hpp"

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

int main()
{
    constexpr int WIDTH = 4096;
    constexpr int HEIGHT = 4096;

    std::cout << "HIPVision - Day 2\n";

    Image input(WIDTH, HEIGHT, 3);

    //Generate synthetic RGB image.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for(auto& pixel : input.data)
        pixel = static_cast<uint8_t>(dist(rng));

    std::cout << "Image: " << WIDTH << "x" << HEIGHT << std::endl;

    // CPU benchmark
    auto cpuStart = Clock::now();
    Image cpuOutput = grayscaleCPU(input);
    auto cpuEnd = Clock::now();
    double cpuTime = elapsedMs(cpuStart, cpuEnd);

    // CUDA benchmark
    auto cudaStart = Clock::now();
    Image cudaOutput = grayscaleCUDA(input);
    auto cudaEnd = Clock::now();
    double cudaTime = elapsedMs(cudaStart, cudaEnd);

    std::cout << "\nCPU Time: " << cpuTime << " ms\n";
    std::cout << "CUDA Time: " << cudaTime << " ms\n";

    if (compareImages(cpuOutput, cudaOutput))
        std::cout << "CPU and CUDA outputs match!\n";
    else
        std::cout << "CPU and CUDA outputs do NOT match!\n";

    if (!compareImages(cpuOutput, cudaOutput))
    {
        size_t mismatchCount = 0;
        for (size_t i = 0; i < cpuOutput.data.size(); ++i)
        {
            if (cpuOutput.data[i] != cudaOutput.data[i])
                ++mismatchCount;
        }
        std::cerr << "Number of mismatched pixels: " << mismatchCount << "\n";
    }

    std::cout << "First CPU pixel: " << static_cast<int>(cpuOutput.data[0]) << "\n";
    std::cout << "First CUDA pixel: " << static_cast<int>(cudaOutput.data[0]) << "\n";

    return 0;
}