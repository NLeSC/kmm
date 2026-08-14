#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/device.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

DeviceMemoryAllocator::DeviceMemoryAllocator(CUcontext context) :
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
    spdlog::trace("allocate {} bytes of device memory (addr: {})", layout.size_in_bytes, *addr_out);
    return AllocResult::Success;
}

void DeviceMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of device memory (addr: {})", layout.size_in_bytes, addr);
    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFree(CUdeviceptr(addr)));
}

}  // namespace kmm
