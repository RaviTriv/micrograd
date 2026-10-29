micrograd_module(ckpt DEPS micrograd::core)

micrograd_test(micrograd_ckpt_test
    SOURCES ckpt_test.cpp
    DEPS micrograd::ckpt
)
