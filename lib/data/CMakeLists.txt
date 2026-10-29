micrograd_module(data DEPS micrograd::core)

micrograd_test(micrograd_data_test
    SOURCES data_test.cpp
    DEPS micrograd::data
)
