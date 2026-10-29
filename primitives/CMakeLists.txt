micrograd_module(primitives DEPS micrograd::core micrograd::kernels)

micrograd_test(micrograd_primitives_test
    SOURCES primitives_test.cpp
    DEPS micrograd::primitives
)
