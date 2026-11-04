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

    float getH2DTimeMs() const { return h2dTimeMs; }
    float getD2HTimeMs() const { return d2hTimeMs; }
    float getKernelTimeMs() const { return kernelTimeMs; }

private:
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    size_t maxPixels = 0;

    float h2dTimeMs = 0.0f;
    float d2hTimeMs = 0.0f;
    float kernelTimeMs = 0.0f;
};