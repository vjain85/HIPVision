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

add_executable(hipvision
    src/main.cpp
    src/cpu/grayscale.cpp
)

add_executable(grayscale_test
    src/cuda/grayscale_test.cu
)

target_include_directories(hipvision
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(hipvision PRIVATE /W4)
else()
    target_compile_options(hipvision PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()