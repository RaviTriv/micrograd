micrograd_module(tools DEPS micrograd::gpt micrograd::optim micrograd::ckpt micrograd::data micrograd::obs)

micrograd_test(micrograd_tools_test
    SOURCES tools_test.cpp
    DEPS micrograd::tools
)

if(MICROGRAD_CUDA)
    add_library(micrograd_cuda_smoke STATIC cuda_smoke.cu)
    target_link_libraries(micrograd_cuda_smoke PUBLIC micrograd::options PRIVATE micrograd::cuda)

    micrograd_test(micrograd_cuda_smoke_test
        SOURCES cuda_smoke_test.cpp
        DEPS micrograd_cuda_smoke
    )
endif()
