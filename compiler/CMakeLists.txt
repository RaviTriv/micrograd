micrograd_module(compiler DEPS micrograd::core micrograd::runtime micrograd::kernels)

micrograd_test(micrograd_compiler_test
    SOURCES compiler_test.cpp
    DEPS micrograd::compiler
)
