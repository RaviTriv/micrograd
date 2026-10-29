micrograd_module(transforms DEPS micrograd::core micrograd::compiler)

micrograd_test(micrograd_transforms_test
    SOURCES transforms_test.cpp
    DEPS micrograd::transforms
)
