#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/system.hpp"

namespace kmm {

AllocResult SystemAllocator::allocate(BufferLayout layout, void** addr_out) {
    *addr_out = malloc(layout.size_in_bytes);

    if (*addr_out == nullptr) {
        return AllocResult::ErrorOutOfMemory;
    }

    spdlog::trace("allocate {} bytes of system memory (addr: {})", layout.size_in_bytes, *addr_out);
    return AllocResult::Success;
}

void SystemAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of system memory (addr: {})", layout.size_in_bytes, addr);
    free(addr);
}

}  // namespace kmm