# Embeddings kernel module (optional C++ acceleration)

set(EMBEDDINGS_HEADERS
    include/max_llm/embeddings/embeddings.h
)

set(EMBEDDINGS_SOURCES
    src/embeddings_placeholder.cpp
)

# Create library (optional component)
add_library(max_llm_embeddings_kernels SHARED
    ${EMBEDDINGS_SOURCES} ${EMBEDDINGS_HEADERS}
)

# Link with core and CUDA
target_link_libraries(max_llm_embeddings_kernels
    PUBLIC
        max_llm_core
        ${CUDA_LIBRARIES}
)

# Include directories
target_include_directories(max_llm_embeddings_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
)

set_target_properties(max_llm_embeddings_kernels PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Tests
add_subdirectory(tests)
