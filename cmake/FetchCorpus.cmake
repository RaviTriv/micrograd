if(NOT DEFINED CORPUS_DATA_DIR)
    message(FATAL_ERROR "CORPUS_DATA_DIR must be set")
endif()

set(CORPUS_URL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
set(CORPUS_FILE "${CORPUS_DATA_DIR}/input.txt")

file(MAKE_DIRECTORY "${CORPUS_DATA_DIR}")

if(NOT EXISTS "${CORPUS_FILE}")
    message(STATUS "Downloading Shakespeare corpus")
    file(DOWNLOAD
        "${CORPUS_URL}"
        "${CORPUS_FILE}"
        STATUS download_status
        TLS_VERIFY ON
    )
    list(GET download_status 0 code)
    if(NOT code EQUAL 0)
        list(GET download_status 1 reason)
        file(REMOVE "${CORPUS_FILE}")
        message(FATAL_ERROR
            "Failed to download the Shakespeare corpus: ${reason}\n"
            "Configure with -DFETCH_CORPUS=OFF to build without it.")
    endif()
endif()
