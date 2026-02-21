# Core library tests

# Find GTest (optional)
if(MAX_LLM_ENABLE_GTEST)
    find_package(GTest CONFIG)
endif()

if(MAX_LLM_ENABLE_GTEST AND GTest_FOUND)
    add_executable(test_core test_version.cpp)

    target_link_libraries(test_core
        PRIVATE
            max_llm_core
            GTest::gtest
            GTest::gtest_main
    )

    add_test(
        NAME core_tests
        COMMAND test_core
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    message(STATUS "Core tests enabled")
else()
    message(STATUS "GTest disabled or not found; skipping core tests")
endif()

if(CMAKE_CUDA_COMPILER)
    add_executable(test_cuda_smoke test_cuda_smoke.cu)

    target_link_libraries(test_cuda_smoke
        PRIVATE
            max_llm_core
            ${CUDA_LIBRARIES}
    )

    target_include_directories(test_cuda_smoke
        PRIVATE
            ${CUDA_INCLUDE_DIRS}
    )

    add_test(
        NAME cuda_smoke
        COMMAND test_cuda_smoke
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    message(STATUS "CUDA smoke test enabled")
else()
    message(STATUS "CUDA compiler not available; skipping CUDA smoke test")
endif()
