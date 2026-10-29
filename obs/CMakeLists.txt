micrograd_module(obs)

micrograd_test(micrograd_obs_test
    SOURCES obs_test.cpp
    DEPS micrograd::obs
)
