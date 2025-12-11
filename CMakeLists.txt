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

add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion -Wsign-conversion
)

if(APPLE)
    option(ENABLE_METAL "Metal GPU acceleration" ON)
    if(ENABLE_METAL)
        add_compile_definitions(MICROGRAD_METAL_ENABLED)
    endif()
endif()

add_library(micrograd
    src/Tensor.cpp
    src/Utils.cpp
    src/nn.cpp
    src/mnist.cpp
    src/metal/MetalContext.cpp
    src/metal/Kernels.cpp
)

target_include_directories(micrograd PUBLIC include)

if(APPLE AND ENABLE_METAL)
    target_include_directories(micrograd PUBLIC external/metal-cpp)
    target_link_libraries(micrograd PUBLIC
        "-framework Metal"
        "-framework Foundation"
    )
endif()

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE micrograd)