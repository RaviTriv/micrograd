micrograd_module(runtime DEPS micrograd::kernels micrograd::obs)

micrograd_test(micrograd_runtime_test
    SOURCES runtime_test.cpp
    DEPS micrograd::runtime
)
