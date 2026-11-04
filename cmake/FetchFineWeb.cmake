if(NOT DEFINED FINEWEB_DATA_DIR)
    message(FATAL_ERROR "FINEWEB_DATA_DIR must be set")
endif()
if(NOT DEFINED FINEWEB_TRAIN_SHARD_COUNT)
    message(FATAL_ERROR "FINEWEB_TRAIN_SHARD_COUNT must be set")
endif()

set(FINEWEB_BASE_URL
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main")

file(MAKE_DIRECTORY "${FINEWEB_DATA_DIR}")

function(fineweb_download remote_name local_name)
    set(local_path "${FINEWEB_DATA_DIR}/${local_name}")
    if(EXISTS "${local_path}")
        return()
    endif()

    message(STATUS "Downloading FineWeb shard: ${remote_name}")
    file(DOWNLOAD
        "${FINEWEB_BASE_URL}/${remote_name}"
        "${local_path}"
        STATUS download_status
        TLS_VERIFY ON
    )
    list(GET download_status 0 code)
    if(NOT code EQUAL 0)
        list(GET download_status 1 reason)
        file(REMOVE "${local_path}")
        message(FATAL_ERROR
            "Failed to download ${remote_name}: ${reason}\n"
            "Configure with -DFETCH_FINEWEB=OFF to build without it.")
    endif()
endfunction()

fineweb_download("edu_fineweb_val_000000.bin" "edufineweb_val_000000.bin")

foreach(shard RANGE 1 ${FINEWEB_TRAIN_SHARD_COUNT})
    set(padded "${shard}")
    string(LENGTH "${padded}" digits)
    while(digits LESS 6)
        set(padded "0${padded}")
        math(EXPR digits "${digits} + 1")
    endwhile()
    fineweb_download("edu_fineweb_train_${padded}.bin"
                     "edufineweb_train_${shard}.bin")
endforeach()
