#include <utility>

#include "kmm/runtime/allocators/pinned.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(CUcontext context, size_t max_bytes) :
    SyncAllocator(max_bytes),
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
    } else {
        KMM_CUDA_CHECK(result);
        return AllocResult::Success;
    }
}

void PinnedMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFreeHost(addr));
}

}  // namespace kmm
