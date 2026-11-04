if(NOT DEFINED MNIST_DATA_DIR)
    message(FATAL_ERROR "MNIST_DATA_DIR must be set")
endif()

set(MNIST_BASE_URL "https://ossci-datasets.s3.amazonaws.com/mnist")

set(MNIST_FILES
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
)

set(MNIST_SHA256_train-images-idx3-ubyte
    440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609)
set(MNIST_SHA256_train-labels-idx1-ubyte
    3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c)
set(MNIST_SHA256_t10k-images-idx3-ubyte
    8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6)
set(MNIST_SHA256_t10k-labels-idx1-ubyte
    f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6)

find_program(GZIP_EXECUTABLE gzip)
if(NOT GZIP_EXECUTABLE)
    message(FATAL_ERROR
        "gzip not found, needed to unpack MNIST. "
        "Install gzip, or configure with -DFETCH_MNIST=OFF to skip the demo dataset.")
endif()

file(MAKE_DIRECTORY "${MNIST_DATA_DIR}")

foreach(name IN LISTS MNIST_FILES)
    set(sha "${MNIST_SHA256_${name}}")

    if(EXISTS "${MNIST_DATA_DIR}/${name}")
        continue()
    endif()

    message(STATUS "Downloading MNIST: ${name}")
    file(DOWNLOAD
        "${MNIST_BASE_URL}/${name}.gz"
        "${MNIST_DATA_DIR}/${name}.gz"
        EXPECTED_HASH SHA256=${sha}
        STATUS download_status
        TLS_VERIFY ON
    )
    list(GET download_status 0 code)
    if(NOT code EQUAL 0)
        list(GET download_status 1 reason)
        file(REMOVE "${MNIST_DATA_DIR}/${name}.gz")
        message(FATAL_ERROR
            "Failed to download ${name}.gz: ${reason}\n"
            "Configure with -DFETCH_MNIST=OFF to build without the demo dataset.")
    endif()

    execute_process(
        COMMAND "${GZIP_EXECUTABLE}" -df "${MNIST_DATA_DIR}/${name}.gz"
        RESULT_VARIABLE gzip_result
        ERROR_VARIABLE gzip_error
    )
    if(NOT gzip_result EQUAL 0)
        message(FATAL_ERROR "Failed to unpack ${name}.gz: ${gzip_error}")
    endif()
endforeach()
