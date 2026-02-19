/**
 * @file test_version.cpp
 * @brief Tests for version info and core utilities
 */

#include <gtest/gtest.h>

#include "max_llm/common.h"
#include "max_llm/version.h"

TEST(VersionTest, MacrosAreDefined) {
    EXPECT_EQ(MAX_LLM_VERSION_MAJOR, 0);
    EXPECT_EQ(MAX_LLM_VERSION_MINOR, 1);
    EXPECT_EQ(MAX_LLM_VERSION_PATCH, 0);
}

TEST(VersionTest, VersionStringMatches) {
    EXPECT_STREQ(MAX_LLM_VERSION_STRING, "0.1.0");
}

TEST(VersionTest, GetVersionReturnsCorrectValues) {
    auto info = max_llm::get_version();
    EXPECT_EQ(info.major, MAX_LLM_VERSION_MAJOR);
    EXPECT_EQ(info.minor, MAX_LLM_VERSION_MINOR);
    EXPECT_EQ(info.patch, MAX_LLM_VERSION_PATCH);
    EXPECT_NE(info.build_type, nullptr);
}
