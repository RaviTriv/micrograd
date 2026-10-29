if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
enable_language(CUDA)

find_package(CUDAToolkit 12 REQUIRED)

find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "MICROGRAD_CUDA=ON needs cuDNN; use docker/cuda.Dockerfile")
endif()

add_library(micrograd_cuda INTERFACE)
add_library(micrograd::cuda ALIAS micrograd_cuda)
target_include_directories(micrograd_cuda SYSTEM INTERFACE ${CUDNN_INCLUDE_DIR})
target_link_libraries(micrograd_cuda INTERFACE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    ${CUDNN_LIBRARY}
)
