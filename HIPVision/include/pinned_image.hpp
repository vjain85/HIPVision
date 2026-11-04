#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

class PinnedImage
{
public:

    int width;
    int height;
    int channels;

    PinnedImage(int w, int h, int c)
        : width(w), height(h), channels(c)
    {
        if (w <= 0 || h <= 0 || c <= 0)
        {
            throw std::invalid_argument("Width, height, and channels must be positive integers.");
        }

        const size_t bytes = static_cast<size_t>(w) * h * c;
        cudaError_t err = cudaMallocHost(&data, bytes);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate pinned host memory: " + std::string(cudaGetErrorString(err)));
        }
        byteCount = bytes;
    }

    ~PinnedImage()
    {
        if (data)
        {
            cudaFreeHost(data);
            data = nullptr;
        }
    }

    // Prevent accidental shallow copies
    PinnedImage(const PinnedImage&) = delete;
    PinnedImage& operator=(const PinnedImage&) = delete;

    // Allow ownership transfer through move semantics
    PinnedImage(PinnedImage&& other) noexcept
        : width(other.width), height(other.height), channels(other.channels), data(other.data), byteCount(other.byteCount)
    {
        other.data = nullptr;
        other.byteCount = 0;
    }

    PinnedImage& operator=(PinnedImage&& other) noexcept
    {
        if (this != &other)
        {
            // Free existing resources
            if (data)
            {
                cudaFreeHost(data);
            }

            // Transfer ownership
            width = other.width;
            height = other.height;
            channels = other.channels;
            data = other.data;
            byteCount = other.byteCount;

            // Nullify the source
            other.data = nullptr;
            other.byteCount = 0;
        }
        return *this;
    }

    size_t pixelCount() const
    {
        return static_cast<size_t>(width) * height;
    }

    size_t sizeInBytes() const
    {
        return byteCount;
    }

    uint8_t* pixels()
    {
        return data;
    }

    const uint8_t* pixels() const
    {
        return data;
    }
private:
    uint8_t* data = nullptr;
    size_t byteCount = 0;
};
