add_library(micrograd_options INTERFACE)
add_library(micrograd::options ALIAS micrograd_options)

target_compile_features(micrograd_options INTERFACE cxx_std_20)
target_compile_options(micrograd_options INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wsign-conversion;-Werror>"
)

if(MICROGRAD_CUDA)
    target_compile_definitions(micrograd_options INTERFACE MICROGRAD_CUDA=1)
endif()
