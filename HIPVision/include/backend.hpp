#pragma once

#include "image.hpp"

Image grayscaleCPU(const Image& input);
Image grayscaleCUDA(const Image& input);
