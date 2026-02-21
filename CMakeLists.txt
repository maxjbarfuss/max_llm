cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDACXX})
        set(CMAKE_CUDA_COMPILER $ENV{CUDACXX} CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

project(
    max_llm
    VERSION 0.1.0
    LANGUAGES C CXX CUDA
    DESCRIPTION "Hybrid LLM with MLA-Transformer-MoE-RNN"
)

option(MAX_LLM_ENABLE_GTEST "Enable GTest-based C++ tests" OFF)

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
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-Wall>
        $<$<COMPILE_LANGUAGE:C>:-Wextra>
        $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:C>:-Werror>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-O3>
        $<$<COMPILE_LANGUAGE:C>:-march=native>
        $<$<COMPILE_LANGUAGE:C>:-g>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
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

# Enable testing
include(CTest)
enable_testing()

# Add subdirectories for vertical slices
message(STATUS "Building core infrastructure...")
add_subdirectory(src/core)

message(STATUS "Building feature kernels...")
add_subdirectory(src/models/embeddings/kernels)
add_subdirectory(src/models/position/kernels)
add_subdirectory(src/models/attention/kernels)
add_subdirectory(src/models/moe/kernels)
add_subdirectory(src/models/rnn/kernels)

# Print build configuration summary
message(STATUS "Max LLM Build Configuration")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  C Standard: ${CMAKE_C_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "  CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "  Build dir: ${CMAKE_BINARY_DIR}")
