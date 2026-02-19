# MoE kernel module

set(MOE_HEADERS
    include/max_llm/moe/moe.h
)

set(MOE_SOURCES
    src/moe_placeholder.cpp
)

add_library(max_llm_moe_kernels SHARED
    ${MOE_SOURCES} ${MOE_HEADERS}
)

target_link_libraries(max_llm_moe_kernels
    PUBLIC
        max_llm_core
        ${CUDA_LIBRARIES}
)

target_include_directories(max_llm_moe_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(max_llm_moe_kernels PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Tests
add_subdirectory(tests)
