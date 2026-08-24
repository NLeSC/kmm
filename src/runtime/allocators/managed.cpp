#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/managed.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

ManagedMemoryAllocator::ManagedMemoryAllocator(GPUContext context) :
    m_context(context) {}

AllocResult ManagedMemoryAllocator::allocate(BufferLayout layout, void** addr_out) {
    GPUContextGuard guard {m_context};
    GPUDeviceptr ptr;
    GPUResult result = gpuMemAllocManaged(&ptr, layout.size_in_bytes);

    if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    *addr_out = (void*)ptr;
    spdlog::trace("allocate {} bytes of managed memory (addr: {})", layout.size_in_bytes, *addr_out);
    return AllocResult::Success;
}

void ManagedMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of managed memory (addr: {})", layout.size_in_bytes, addr);

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemFree(GPUDeviceptr(addr)));
}

}  // namespace kmm
