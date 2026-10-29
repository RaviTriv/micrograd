micrograd_module(gpt DEPS micrograd::nn)

micrograd_test(micrograd_gpt_test
    SOURCES gpt_test.cpp
    DEPS micrograd::gpt
)
