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

add_library(micrograd
    src/Tensor.cpp
    src/Utils.cpp
    src/nn.cpp
    src/mnist.cpp
)

target_include_directories(micrograd PUBLIC include)
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE micrograd)