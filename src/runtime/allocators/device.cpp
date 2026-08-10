#include <utility>

#include "kmm/runtime/allocators/device.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

DeviceMemoryAllocator::DeviceMemoryAllocator(CUcontext context, size_t max_bytes) :
    SyncAllocator(max_bytes),
    m_context(context) {}

AllocResult DeviceMemoryAllocator::allocate(BufferLayout layout, void** addr_out) {
    CUDAContextGuard guard {m_context};
    CUdeviceptr ptr;
    CUresult result = cuMemAlloc(&ptr, layout.size_in_bytes);

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_CUDA_CHECK(result);
    *addr_out = (void*)ptr;
    return AllocResult::Success;
}

void DeviceMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFree(CUdeviceptr(addr)));
}

}  // namespace kmm
