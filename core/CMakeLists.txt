micrograd_module(core DEPS micrograd::runtime)

micrograd_test(micrograd_core_test
    SOURCES core_test.cpp
    DEPS micrograd::core
)
