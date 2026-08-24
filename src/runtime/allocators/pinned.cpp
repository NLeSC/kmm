#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/pinned.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(GPUContext context) :
    m_context(context) {}

AllocResult PinnedMemoryAllocator::allocate(BufferLayout layout, void** addr_out) {
    GPUContextGuard guard {m_context};
    GPUResult result = gpuMemHostAlloc(addr_out, layout.size_in_bytes);

    if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    spdlog::trace("allocate {} bytes of pinned memory (addr: {})", layout.size_in_bytes, *addr_out);
    return AllocResult::Success;
}

void PinnedMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of pinned memory (addr: {})", layout.size_in_bytes, addr);

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemFreeHost(addr));
}

}  // namespace kmm
