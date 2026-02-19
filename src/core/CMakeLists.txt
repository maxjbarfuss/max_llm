# Shared core infrastructure for Max LLM

set(CORE_SOURCES
    src/common.cpp
)

set(CORE_HEADERS
    include/max_llm/common.h
    include/max_llm/version.h
)

# Create library
add_library(max_llm_core SHARED ${CORE_SOURCES} ${CORE_HEADERS})

# Link with Python and CUDA
target_link_libraries(max_llm_core
    PUBLIC
        ${Python3_LIBRARIES}
        ${CUDA_LIBRARIES}
)

# Set include directories
target_include_directories(max_llm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Python3_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

# Set library properties
set_target_properties(max_llm_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Tests
add_subdirectory(tests)
