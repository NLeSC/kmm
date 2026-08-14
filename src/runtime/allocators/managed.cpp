#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/managed.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

ManagedMemoryAllocator::ManagedMemoryAllocator(CUcontext context) :
    m_context(context) {}

AllocResult ManagedMemoryAllocator::allocate(BufferLayout layout, void** addr_out) {
    CUDAContextGuard guard {m_context};
    CUdeviceptr ptr;
    CUresult result = cuMemAllocManaged(&ptr, layout.size_in_bytes, CU_MEM_ATTACH_GLOBAL);

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_CUDA_CHECK(result);
    *addr_out = (void*)ptr;
    spdlog::trace("allocate {} bytes of managed memory (addr: {})", layout.size_in_bytes, *addr_out);
    return AllocResult::Success;
}

void ManagedMemoryAllocator::deallocate(void* addr, BufferLayout layout) {
    spdlog::trace("deallocate {} bytes of managed memory (addr: {})", layout.size_in_bytes, addr);

    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFree(CUdeviceptr(addr)));
}

}  // namespace kmm
