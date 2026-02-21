# MoE tests

if(MAX_LLM_ENABLE_GTEST)
    find_package(GTest CONFIG)
endif()

if(MAX_LLM_ENABLE_GTEST AND GTest_FOUND)
    add_executable(test_moe test_placeholder.cpp)

    target_link_libraries(test_moe
        PRIVATE
            max_llm_moe_kernels
            max_llm_core
            GTest::gtest
            GTest::gtest_main
    )

    add_test(
        NAME moe_tests
        COMMAND test_moe
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    message(STATUS "MoE tests enabled")
else()
    message(STATUS "GTest disabled or not found; skipping MoE tests")
endif()
