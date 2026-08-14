#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/pinned.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(CUcontext context) :
    m_context(context) {}

AllocResult PinnedMemoryAllocator::allocate(BufferLayout layout, void** addr_out) {
    CUDAContextGuard guard {m_context};
    CUresult result = cuMemHostAlloc(  //
        addr_out,
        layout.size_in_bytes,
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP
    );

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_CUDA_CHECK(result);
    spdlog::trace("allocate {} bytes of pinned memory (addr: {})", layout.size_in_bytes, *addr_out);
    return AllocResult::Success;
}

void PinnedMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of pinned memory (addr: {})", layout.size_in_bytes, addr);

    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFreeHost(addr));
}

}  // namespace kmm
