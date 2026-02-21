# RNN tests

if(MAX_LLM_ENABLE_GTEST)
    find_package(GTest CONFIG)
endif()

if(MAX_LLM_ENABLE_GTEST AND GTest_FOUND)
    add_executable(test_rnn test_placeholder.cpp)

    target_link_libraries(test_rnn
        PRIVATE
            max_llm_rnn_kernels
            max_llm_core
            GTest::gtest
            GTest::gtest_main
    )

    add_test(
        NAME rnn_tests
        COMMAND test_rnn
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    message(STATUS "RNN tests enabled")
else()
    message(STATUS "GTest disabled or not found; skipping RNN tests")
endif()
