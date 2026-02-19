# Position (RoPE) kernel module

set(POSITION_HEADERS
    include/max_llm/position/rope.h
)

set(POSITION_SOURCES
    src/rope_placeholder.cpp
)

add_library(max_llm_position_kernels SHARED
    ${POSITION_SOURCES} ${POSITION_HEADERS}
)

target_link_libraries(max_llm_position_kernels
    PUBLIC
        max_llm_core
        ${CUDA_LIBRARIES}
)

target_include_directories(max_llm_position_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(max_llm_position_kernels PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Tests
add_subdirectory(tests)
