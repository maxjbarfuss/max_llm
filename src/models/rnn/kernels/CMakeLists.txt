# RNN kernel module

set(RNN_HEADERS
    include/max_llm/rnn/gru.h
)

set(RNN_SOURCES
    src/gru_placeholder.cpp
)

add_library(max_llm_rnn_kernels SHARED
    ${RNN_SOURCES} ${RNN_HEADERS}
)

target_link_libraries(max_llm_rnn_kernels
    PUBLIC
        max_llm_core
        ${CUDA_LIBRARIES}
)

target_include_directories(max_llm_rnn_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(max_llm_rnn_kernels PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Tests
add_subdirectory(tests)
