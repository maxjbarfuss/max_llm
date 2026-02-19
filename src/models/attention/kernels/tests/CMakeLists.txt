# Attention tests

find_package(GTest CONFIG)

if(GTest_FOUND)
    add_executable(test_attention test_placeholder.cpp)

    target_link_libraries(test_attention
        PRIVATE
            max_llm_attention_kernels
            max_llm_core
            GTest::gtest
            GTest::gtest_main
    )

    add_test(
        NAME attention_tests
        COMMAND test_attention
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    message(STATUS "Attention tests enabled")
else()
    message(STATUS "GTest not found; skipping attention tests")
endif()
