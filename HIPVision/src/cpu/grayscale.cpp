#include "backend.hpp"

#include <cassert>
#include <iostream>
#include <cstdint>

Image grayscaleCPU(const Image& input)
{
    assert(input.channels == 3);
    assert(input.data.size() == input.pixelCount() * input.channels);

    Image output (input.width, input.height, 1);
    
    const uint8_t* src = input.pixels();
    uint8_t* dst = output.pixels();

    const size_t totalPixels = input.pixelCount();

    for (size_t i = 0; i < totalPixels; ++i)
    {
        const size_t rgbIdx = i * input.channels;

        const uint8_t r = src[rgbIdx];
        const uint8_t g = src[rgbIdx + 1];
        const uint8_t b = src[rgbIdx + 2];

        const uint8_t gray = static_cast<uint8_t>(299u * r + 587u * g + 114u * b) / 1000u;

        dst[i] = gray;
    }
    return output;
}