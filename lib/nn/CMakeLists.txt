micrograd_module(nn DEPS micrograd::transforms micrograd::primitives)

micrograd_test(micrograd_nn_test
    SOURCES nn_test.cpp
    DEPS micrograd::nn
)
