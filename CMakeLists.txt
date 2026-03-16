cmake_minimum_required(VERSION 3.16) 
project(DeepModel LANGUAGES CXX) 
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) 


option(ENABLE_CUDA "CUDA" OFF)
option(BUILD_EXAMPLES "Examples" OFF)
option(BUILD_BENCHMARK "Benchmark" OFF)


add_subdirectory(src)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()







