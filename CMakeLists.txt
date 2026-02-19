cmake_minimum_required(VERSION 3.20)
project(
    max_llm
    VERSION 0.1.0
    LANGUAGES C CXX CUDA
    DESCRIPTION "Hybrid LLM with MLA-Transformer-MoE-RNN"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Project structure
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# Compiler flags for optimization and debugging
if(MSVC)
    # Windows MSVC
    add_compile_options(/W4 /WX)
    add_compile_options(/O2)
else()
    # GCC/Clang
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options(-O3 -march=native)

    # Debug symbols
    add_compile_options(-g)
endif()

# CUDA configuration
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find required packages
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(CUDA 12.1 REQUIRED)

# Optional: Find pybind11 for Python binding support
find_package(pybind11 CONFIG)

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Add subdirectories for vertical slices
message(STATUS "Building core infrastructure...")
add_subdirectory(src/core)

message(STATUS "Building feature kernels...")
add_subdirectory(src/models/embeddings/kernels)
add_subdirectory(src/models/position/kernels)
add_subdirectory(src/models/attention/kernels)
add_subdirectory(src/models/moe/kernels)
add_subdirectory(src/models/rnn/kernels)

# Enable testing
enable_testing()

# Print build configuration summary
message(STATUS "Max LLM Build Configuration")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  C Standard: ${CMAKE_C_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "  CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "  Build dir: ${CMAKE_BINARY_DIR}")
