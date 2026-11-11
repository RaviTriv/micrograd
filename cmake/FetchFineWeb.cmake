if(NOT DEFINED FINEWEB_DATA_DIR)
    message(FATAL_ERROR "FINEWEB_DATA_DIR must be set")
endif()
if(NOT DEFINED FINEWEB_TRAIN_SHARD_COUNT)
    message(FATAL_ERROR "FINEWEB_TRAIN_SHARD_COUNT must be set")
endif()

set(FINEWEB_BASE_URL
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main")
set(FINEWEB_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/fineweb.sha256")

file(STRINGS "${FINEWEB_MANIFEST}" manifest_lines)
set(train_available 0)
foreach(line IN LISTS manifest_lines)
    if(line MATCHES "^([0-9a-f]+)  (.+)$")
        set("fineweb_sha256_${CMAKE_MATCH_2}" "${CMAKE_MATCH_1}")
        if(CMAKE_MATCH_2 MATCHES "^edu_fineweb_train_")
            math(EXPR train_available "${train_available} + 1")
        endif()
    endif()
endforeach()
if(FINEWEB_TRAIN_SHARD_COUNT GREATER train_available)
    message(FATAL_ERROR
        "FINEWEB_TRAIN_SHARD_COUNT is ${FINEWEB_TRAIN_SHARD_COUNT} but "
        "${FINEWEB_MANIFEST} lists ${train_available} train shards")
endif()

file(MAKE_DIRECTORY "${FINEWEB_DATA_DIR}")

function(fineweb_download remote_name local_name)
    set(expected "${fineweb_sha256_${remote_name}}")
    if(expected STREQUAL "")
        message(FATAL_ERROR "${remote_name} is not listed in ${FINEWEB_MANIFEST}")
    endif()

    set(local_path "${FINEWEB_DATA_DIR}/${local_name}")
    if(EXISTS "${local_path}")
        file(SHA256 "${local_path}" actual)
        if(actual STREQUAL expected)
            file(TOUCH_NOCREATE "${local_path}")
            return()
        endif()
        message(STATUS "Removing ${local_path}: SHA-256 mismatch")
        file(REMOVE "${local_path}")
    endif()

    message(STATUS "Downloading FineWeb shard: ${remote_name}")
    file(DOWNLOAD
        "${FINEWEB_BASE_URL}/${remote_name}"
        "${local_path}.part"
        STATUS download_status
        TLS_VERIFY ON
    )
    list(GET download_status 0 code)
    if(NOT code EQUAL 0)
        list(GET download_status 1 reason)
        file(REMOVE "${local_path}.part")
        message(FATAL_ERROR
            "Failed to download ${remote_name}: ${reason}\n"
            "Configure with -DFETCH_FINEWEB=OFF to build without it.")
    endif()

    file(SHA256 "${local_path}.part" actual)
    if(NOT actual STREQUAL expected)
        file(REMOVE "${local_path}.part")
        message(FATAL_ERROR
            "SHA-256 mismatch for ${remote_name}\n"
            "  expected ${expected}\n"
            "  actual   ${actual}\n"
            "The file was removed.")
    endif()
    file(RENAME "${local_path}.part" "${local_path}")
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
