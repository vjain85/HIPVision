#pragma once

#include <cstdint>
#include <vector>

class Image
{
    public:

    int width;
    int height;
    int channels;

    std::vector<uint8_t> data;

    Image (int w, int h, int c)
        : width(w),
          height(h),
          channels(c),
          data(static_cast<size_t>(w) * h * c, 0)
    {
    
    }  

    size_t pixelCount() const
    {
        return static_cast<size_t>(width) * height;
    }

    uint8_t* pixels()
    {
        return data.data();
    }

    const uint8_t* pixels() const
    {
        return data.data();
    }
};