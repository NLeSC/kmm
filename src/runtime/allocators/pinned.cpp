#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/pinned.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(g_context_t context) : m_context(context) {}

AllocResult PinnedMemoryAllocator::allocate(BufferLayout layout, void** addr_out) {
    GPUContextGuard guard {m_context};
    g_result_t result = g_mem_host_alloc(
        addr_out,
        layout.size_in_bytes,
        G_MEMHOSTALLOC_PORTABLE | G_MEMHOSTALLOC_DEVICEMAP
    );

    if (result == G_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    spdlog::trace("allocate {} bytes of pinned memory (addr: {})", layout.size_in_bytes, *addr_out);
    return AllocResult::Success;
}

void PinnedMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of pinned memory (addr: {})", layout.size_in_bytes, addr);

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_mem_free_host(addr));
}

}  // namespace kmm
