# Attention kernel module

set(ATTENTION_HEADERS
    include/max_llm/attention/mla.h
)

set(ATTENTION_SOURCES
    src/mla_placeholder.cpp
)

add_library(max_llm_attention_kernels SHARED
    ${ATTENTION_SOURCES} ${ATTENTION_HEADERS}
)

target_link_libraries(max_llm_attention_kernels
    PUBLIC
        max_llm_core
        ${CUDA_LIBRARIES}
)

target_include_directories(max_llm_attention_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(max_llm_attention_kernels PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Tests
add_subdirectory(tests)
