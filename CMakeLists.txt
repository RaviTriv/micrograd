cmake_minimum_required(VERSION 3.25)
project(micrograd
    VERSION 0.2.0
    LANGUAGES CXX
    DESCRIPTION "A JAX-style autodiff engine in C++ and CUDA"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MICROGRAD_CUDA "Build CUDA code (sm_80 by default)" OFF)
option(MICROGRAD_BUILD_TESTS "Unit tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MicrogradOptions)
include(MicrogradModule)

if(MICROGRAD_CUDA)
    include(MicrogradCuda)
endif()

if(MICROGRAD_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        SYSTEM
        FIND_PACKAGE_ARGS NAMES GTest
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include(GoogleTest)
endif()

add_subdirectory(kernels)
add_subdirectory(obs)
add_subdirectory(runtime)
add_subdirectory(core)
add_subdirectory(primitives)
add_subdirectory(compiler)
add_subdirectory(transforms)
add_subdirectory(lib/nn)
add_subdirectory(lib/optim)
add_subdirectory(lib/ckpt)
add_subdirectory(lib/data)
add_subdirectory(models/gpt)
add_subdirectory(tools)
add_subdirectory(bench)
