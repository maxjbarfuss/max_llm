# Core library tests

# Find GTest (optional)
find_package(GTest CONFIG)

if(GTest_FOUND)
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
    message(STATUS "GTest not found; skipping core tests")
endif()
