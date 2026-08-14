#include "kmm/runtime/identifiers.hpp"

namespace kmm {

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
