micrograd_module(bench DEPS micrograd::gpt micrograd::transforms)

micrograd_test(micrograd_bench_test
    SOURCES bench_test.cpp
    DEPS micrograd::bench
)
