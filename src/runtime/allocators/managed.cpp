#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/managed.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

ManagedMemoryAllocator::ManagedMemoryAllocator(g_context_t context) : m_context(context) {}

AllocResult ManagedMemoryAllocator::allocate(BufferLayout layout, void** addr_out) {
    GPUContextGuard guard {m_context};
    g_device_ptr_t ptr;
    g_result_t result = g_mem_alloc_managed(&ptr, layout.size_in_bytes, G_MEM_ATTACH_GLOBAL);

    if (result == G_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    *addr_out = (void*)ptr;
    spdlog::trace(
        "allocate {} bytes of managed memory (addr: {})",
        layout.size_in_bytes,
        *addr_out
    );
    return AllocResult::Success;
}

void ManagedMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of managed memory (addr: {})", layout.size_in_bytes, addr);

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_mem_free(g_device_ptr_t(addr)));
}

}  // namespace kmm
