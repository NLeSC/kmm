#include "kmm/runtime/allocators/system.hpp"

namespace kmm {

AllocResult SystemAllocator::allocate(BufferLayout layout, void** addr_out) {
    *addr_out = malloc(layout.size_in_bytes);
    return *addr_out != nullptr ? AllocResult::Success : AllocResult::ErrorOutOfMemory;
}

void SystemAllocator::deallocate(void* addr, BufferLayout layout) {
    free(addr);
}

}  // namespace kmm