# Position (RoPE) tests

if(MAX_LLM_ENABLE_GTEST)
    find_package(GTest CONFIG)
endif()

if(MAX_LLM_ENABLE_GTEST AND GTest_FOUND)
    add_executable(test_position test_placeholder.cpp)

    target_link_libraries(test_position
        PRIVATE
            max_llm_position_kernels
            max_llm_core
            GTest::gtest
            GTest::gtest_main
    )

    add_test(
        NAME position_tests
        COMMAND test_position
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    message(STATUS "Position tests enabled")
else()
    message(STATUS "GTest disabled or not found; skipping position tests")
endif()
