#include <stdexcept>

#include "kmm/core/checked_compare.hpp"

namespace kmm {

[[noreturn]] void throw_overflow_exception() {
    throw std::overflow_error("integer overflow occurred");
}

}  // namespace kmm