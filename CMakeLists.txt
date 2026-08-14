cmake_minimum_required(VERSION 3.20)
project(micrograd
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "A small autograd engine"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion -Wsign-conversion
)

if(APPLE)
    option(ENABLE_METAL "Metal GPU acceleration" ON)
endif()

add_library(micrograd
    src/core/Tensor.cpp
    src/core/Autograd.cpp

    src/ops/Activations.cpp
    src/ops/Arithmetic.cpp
    src/ops/Matmul.cpp
    src/ops/Reduction.cpp

    src/NN.cpp
    src/MNIST.cpp
    src/metal/MetalContext.cpp
    src/metal/Kernels.cpp
    src/metal/Dispatch.cpp
    src/metal/TensorMetal.cpp
)

target_include_directories(micrograd PUBLIC include)

if(APPLE AND ENABLE_METAL)
    include(FetchContent)
    FetchContent_Declare(metalcpp
        URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip
        URL_HASH SHA256=0433df1e0ab13c2b0becbd78665071e3fa28381e9714a3fce28a497892b8a184
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(metalcpp)

    target_compile_definitions(micrograd PUBLIC MICROGRAD_METAL_ENABLED)
    target_include_directories(micrograd SYSTEM PUBLIC ${metalcpp_SOURCE_DIR})
    target_link_libraries(micrograd PUBLIC
        "-framework Metal"
        "-framework Foundation"
    )
endif()

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE micrograd)

option(FETCH_MNIST "Download the MNIST dataset" ON)
if(FETCH_MNIST)
    set(MNIST_DATA_DIR "${CMAKE_BINARY_DIR}/mnist-data")
    set(MNIST_DATA_FILES
        "${MNIST_DATA_DIR}/train-images-idx3-ubyte"
        "${MNIST_DATA_DIR}/train-labels-idx1-ubyte"
        "${MNIST_DATA_DIR}/t10k-images-idx3-ubyte"
        "${MNIST_DATA_DIR}/t10k-labels-idx1-ubyte"
    )

    add_custom_command(
        OUTPUT ${MNIST_DATA_FILES}
        COMMAND ${CMAKE_COMMAND} -DMNIST_DATA_DIR=${MNIST_DATA_DIR}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchMNIST.cmake"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchMNIST.cmake"
        COMMENT "Downloading MNIST dataset"
        VERBATIM
    )
    add_custom_target(mnist-data DEPENDS ${MNIST_DATA_FILES})
    add_dependencies(main mnist-data)

    target_compile_definitions(main PRIVATE MNIST_DATA_DIR="${MNIST_DATA_DIR}")
endif()

option(BUILD_TESTS "Unit tests" ON)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(micrograd_tests test/tensor_test.cpp)
    target_link_libraries(micrograd_tests PRIVATE micrograd GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(micrograd_tests)
endif()