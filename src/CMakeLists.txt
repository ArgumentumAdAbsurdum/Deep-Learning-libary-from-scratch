

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(ENABLE_CUDA)

    set(SOURCES 
        dataset.cpp
        neuralnetwork.cpp
        CUDA/kernel.cu
        CUDA/matrixCUDA.cu
        CUDA/activationCUDA.cu
        CUDA/memory_poolCUDA.cu
    )

else()
    set(SOURCES 
        dataset.cpp
        neuralnetwork.cpp
        CPU/matrixCPU.cpp 
        CPU/activationCPU.cpp
        CPU/memory_poolCPU.cpp
    )
endif()


add_library(DeepModel STATIC ${SOURCES})

target_include_directories(DeepModel PUBLIC 
    ${PROJECT_SOURCE_DIR}/headers
)

if(ENABLE_CUDA)
    target_include_directories(DeepModel PUBLIC 
        ${PROJECT_SOURCE_DIR}/headers/CUDA
    )

    target_compile_definitions(DeepModel PUBLIC ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(DeepModel PRIVATE CUDA::cudart)

else()

    target_include_directories(DeepModel PUBLIC 
        ${PROJECT_SOURCE_DIR}/headers/CPU
    )

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(DeepModel PUBLIC OpenMP::OpenMP_CXX)
    endif()

        target_compile_options(DeepModel PRIVATE
            -Ofast
            -march=native
            -mtune=native
            -funroll-loops
            -fomit-frame-pointer
            -flto
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
        )
endif()