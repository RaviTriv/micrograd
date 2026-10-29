micrograd_module(optim DEPS micrograd::transforms micrograd::primitives)

micrograd_test(micrograd_optim_test
    SOURCES optim_test.cpp
    DEPS micrograd::optim
)
