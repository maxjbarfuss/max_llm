/**
 * @file common.cpp
 * @brief Core utility implementations
 */

#include "max_llm/common.h"
#include "max_llm/version.h"

namespace max_llm {

VersionInfo get_version() {
    return VersionInfo{
        .major = MAX_LLM_VERSION_MAJOR,
        .minor = MAX_LLM_VERSION_MINOR,
        .patch = MAX_LLM_VERSION_PATCH,
        .build_type = "development",
    };
}

}  // namespace max_llm
