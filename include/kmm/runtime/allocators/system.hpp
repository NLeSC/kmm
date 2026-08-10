#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class SystemAllocator: public SyncAllocator {
    AllocResult allocate(BufferLayout layout, void** addr_out);
    void deallocate(void* addr, BufferLayout layout);
};

}  // namespace kmm