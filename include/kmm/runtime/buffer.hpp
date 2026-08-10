#pragma once

#include <cstddef>
#include <vector>

#include "kmm/runtime/identifiers.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

enum struct AccessMode {
    /// multiple requests may be active concurrently in multiple memories, but they cannot write
    Read,

    /// only one request may be active in one memory
    ReadWrite,

    /// multiple access will be reduced into one
    Reduce
};

struct BufferLayout {
    BufferLayout repeat(size_t n) {
        size_t remainder = size_in_bytes % alignment;
        size_t padding = remainder != 0 ? alignment - remainder : 0;
        return {(size_in_bytes + padding) * n, alignment};
    }

    template<typename T>
    static BufferLayout for_type(size_t n = 1) {
        return BufferLayout {sizeof(T), alignof(T)}.repeat(n);
    }

    size_t size_in_bytes = 0;
    size_t alignment = 1;
};

struct BufferAccessor {
    MemoryId memory_id = MemoryId::host();
    size_t size_in_bytes;
    bool is_writable;
    void* address;
};

}  // namespace kmm