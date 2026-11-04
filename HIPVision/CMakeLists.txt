cmake_minimum_required(VERSION 3.25)

project(HIPVision
    VERSION 0.1
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(hipvision
    src/main.cpp
    src/cpu/grayscale.cpp
    src/cuda/cuda_grayscale.cu
)

target_include_directories(hipvision PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Compiler warnings for C++
if(MSVC)
    target_compile_options(hipvision PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(hipvision PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()