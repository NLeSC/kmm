#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/core/checked_math.hpp"
#include "kmm/runtime/runtime_config.hpp"

namespace kmm {

static bool is_one_of(const std::string& input, std::initializer_list<const char*> options) {
    for (const char* option : options) {
        if (input == option) {
            return true;
        }
    }

    return false;
}

static size_t parse_byte_size(const char* input) {
    char* end;
    long double result = strtold(input, &end);

    if (end == input) {
        throw std::runtime_error(fmt::format("invalid size in bytes: {}", input));
    }

    auto unit = std::string(end);
    std::transform(unit.begin(), unit.end(), unit.begin(), ::tolower);

    if (is_one_of(unit, {"", "b"})) {
        //
    } else if (is_one_of(unit, {"k", "kb"})) {
        result *= 1000;
    } else if (is_one_of(unit, {"m", "mb"})) {
        result *= 1000'0000;
    } else if (is_one_of(unit, {"g", "gb"})) {
        result *= 1000'000'000;
    } else if (is_one_of(unit, {"t", "tb"})) {
        result *= 1000'000'000'000;
    } else {
        throw std::runtime_error(fmt::format("invalid size in bytes: {}", input));
    }

    if (result < 0 || (result - static_cast<long double>(std::numeric_limits<size_t>::max())) > 0) {
        throw std::runtime_error(fmt::format("invalid size in bytes: {}", input));
    }

    return static_cast<size_t>(result);
}

RuntimeConfig default_config_from_environment() {
    RuntimeConfig config;

    if (auto* s = getenv("KMM_HOST_MEM")) {
        config.host_memory_limit = parse_byte_size(s);
    }

    if (auto* s = getenv("KMM_HOST_BLOCK")) {
        config.host_memory_block_size = parse_byte_size(s);
    }

    if (auto* s = getenv("KMM_DEVICE_MEM")) {
        config.device_memory_limit = parse_byte_size(s);
    }

    if (auto* s = getenv("KMM_DEVICE_BLOCK")) {
        config.device_memory_block_size = parse_byte_size(s);
    }

    if (auto* s = getenv("KMM_DEBUG")) {
        if (is_one_of(s, {"1", "true", "TRUE"})) {
            config.debug_mode = true;
        } else if (is_one_of(s, {"0", "false", "FALSE", ""})) {
            config.debug_mode = false;
        } else {
            throw std::runtime_error(
                fmt::format("invalid value given for KMM_DEBUG: {}", std::string(s))
            );
        }
    }

    if (auto* s = getenv("KMM_LOG_LEVEL")) {
        set_global_log_level(s);
    }

    return config;
}

void set_global_log_level(const std::string& name) {
    spdlog::level::level_enum log_level;

    if (name.empty()) {
        return;  // No log level specified
    } else if (is_one_of(name, {"trace", "TRACE"})) {
        log_level = spdlog::level::level_enum::trace;
    } else if (is_one_of(name, {"debug", "DEBUG"})) {
        log_level = spdlog::level::level_enum::debug;
    } else if (is_one_of(name, {"info", "INFO"})) {
        log_level = spdlog::level::level_enum::info;
    } else if (is_one_of(name, {"warn", "WARN", "warning", "WARNING"})) {
        log_level = spdlog::level::level_enum::warn;
    } else if (is_one_of(name, {"err", "ERR", "error", "ERROR"})) {
        log_level = spdlog::level::level_enum::err;
    } else if (is_one_of(name, {"critical", "CRITICAL"})) {
        log_level = spdlog::level::level_enum::critical;
    } else {
        throw std::runtime_error(fmt::format("invalid log level specified: {}", std::string(name)));
    }

    spdlog::set_level(log_level);
}
}  // namespace kmm