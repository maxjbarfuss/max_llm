#pragma once

/**
 * @file common.h
 * @brief Common utilities and definitions for Max LLM
 */

namespace max_llm {

/**
 * Version information
 */
struct VersionInfo {
    unsigned int major = 0;
    unsigned int minor = 1;
    unsigned int patch = 0;
    const char* build_type = "development";
};

/**
 * Get version information
 */
VersionInfo get_version();

}  // namespace max_llm
