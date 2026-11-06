#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "kmm/runtime/identifiers.hpp"

namespace kmm {

static std::string to_lower(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return result;
}

static MemoryId parse_memory_id(const std::string& name) {
    std::string lower = to_lower(name);

    if (lower == "host" || lower == "cpu") {
        return MemoryId::host();
    }

    size_t sep = lower.find(':');
    std::string prefix = sep == std::string::npos ? lower : lower.substr(0, sep);

    if (prefix == "gpu" || prefix == "cuda" || prefix == "hip" || prefix == "device") {
        if (sep == std::string::npos) {
            return MemoryId::device(DeviceId(0));
        }

        std::string suffix = lower.substr(sep + 1);

        try {
            size_t pos;
            size_t device_index = std::stoul(suffix, &pos);

            if (pos == suffix.size()) {
                return MemoryId::device(DeviceId(device_index));
            }
        } catch (const std::exception&) {
            // fallthrough to error below
        }
    }

    throw std::runtime_error("invalid memory identifier: " + name);
}

MemoryId::MemoryId(const std::string& name) : MemoryId(parse_memory_id(name)) {}

std::ostream& operator<<(std::ostream& stream, const DeviceId& e) {
    return stream << "GPU(" << e.get() << ")";
}

std::ostream& operator<<(std::ostream& stream, const BufferId& e) {
    return stream << "Buffer(" << e.get() << ")";
}

std::ostream& operator<<(std::ostream& stream, const MemoryId& e) {
    if (e.is_host()) {
        return stream << "Host";
    } else {
        return stream << e.as_device();
    }
}

}  // namespace kmm
