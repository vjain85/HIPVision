#include <iostream>
#include <chrono>
#include <random>

#include "image.hpp"

Image grayscaleCPU(const Image& input);

int main()
{
    constexpr int WIDTH = 4096;
    constexpr int HEIGHT = 4096;

    std::cout << "HIPVision - Day1\n";

    Image input(WIDTH, HEIGHT, 3);

    //Generate synthetic RGB image.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for(auto& pixel : input.data)
    {
        pixel = static_cast<uint8_t>(dist(rng));
    }

    std::cout << "Image: " << WIDTH << "x" << HEIGHT << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    Image output = grayscaleCPU(input);

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "CPU grayscale completed\n";

    std::cout << "Execution time: " << elapsed << " ms\n";

    std::cout << "Output pixels: " << output.pixelCount() << "\n";

    std::cout << "First output pixel: " << static_cast<int>(output.data[0]) << "\n";

    return 0;
}