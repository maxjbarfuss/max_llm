# Embeddings tests

if(MAX_LLM_ENABLE_GTEST)
    find_package(GTest CONFIG)
endif()

if(MAX_LLM_ENABLE_GTEST AND GTest_FOUND)
    add_executable(test_embeddings test_placeholder.cpp)

    target_link_libraries(test_embeddings
        PRIVATE
            max_llm_embeddings_kernels
            max_llm_core
            GTest::gtest
            GTest::gtest_main
    )

    add_test(
        NAME embeddings_tests
        COMMAND test_embeddings
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    message(STATUS "Embeddings tests enabled")
else()
    message(STATUS "GTest disabled or not found; skipping embeddings tests")
endif()
