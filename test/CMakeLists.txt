include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(micrograd_tests
    autograd_test.cpp
    serialization_test.cpp
    tensor_core_test.cpp
    training_test.cpp
)
target_link_libraries(micrograd_tests PRIVATE micrograd GTest::gtest_main)
target_compile_options(micrograd_tests PRIVATE -Werror)

include(GoogleTest)
gtest_discover_tests(micrograd_tests PROPERTIES LABELS unit TIMEOUT 60)

function(micrograd_add_check name source)
    add_executable(${name} "${PROJECT_SOURCE_DIR}/${source}")
    target_link_libraries(${name} PRIVATE micrograd_gpt)
    target_compile_options(${name} PRIVATE -Werror)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS check TIMEOUT 120)
endfunction()

micrograd_add_check(gradcheck tools/check/gradcheck.cpp)
micrograd_add_check(bpe_roundtrip tools/check/bpe_roundtrip.cpp)
micrograd_add_check(core_eval tools/eval/core_eval.cpp)
micrograd_add_check(grpo_check tools/check/grpo_check.cpp)
