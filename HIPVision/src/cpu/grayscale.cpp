#include "image.hpp"
#include <cassert>
#include <iostream>

Image grayscaleCPU(const Image& input)
{
    const uint8_t* src = input.pixels();

    std::cout << "Width: " << input.width << '\n';
    std::cout << "Height: " << input.height << '\n';
    std::cout << "Channels: " << input.channels << '\n';
    std::cout << "Source buffer size: " << input.data.size() << '\n';

    const size_t expected = static_cast<size_t>(input.width) * input.height * input.channels;

    std::cout << "Expected buffer size: " << expected << '\n';

    assert(input.data.size() == expected);
    assert(input.channels == 3);

    Image output (input.width, input.height, 1);

    uint8_t* dst = output.pixels();

    const size_t totalPixels = input.pixelCount();

    for (size_t i = 0; i < totalPixels; ++i)
    {
        const size_t rgbIdx = i * input.channels;

        const uint8_t r = src[rgbIdx];
        const uint8_t g = src[rgbIdx + 1];
        const uint8_t b = src[rgbIdx + 2];

        const float gray = 0.299f * r + 0.587f * g + 0.114f * b;

        dst[i] = static_cast<uint8_t>(gray);
    }
    return output;
}