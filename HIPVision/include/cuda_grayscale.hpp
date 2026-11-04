#include "image.hpp"
#include <cstdint>

class CudaGrayscale
{
public:
    CudaGrayscale(size_t maxPixels);
    ~CudaGrayscale();

    CudaGrayscale(const CudaGrayscale&) = delete;
    CudaGrayscale& operator=(const CudaGrayscale&) = delete;

    Image process(const Image& input);

private:
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    size_t maxPixels = 0;
};