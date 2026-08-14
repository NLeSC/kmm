#include <cstring>
#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/device_pool.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

DevicePoolAllocator::DevicePoolAllocator(
    CUcontext context,
    DevicePoolKind kind,
    size_t max_size
) :
    m_context(context),
    m_pool(nullptr),
    m_kind(kind)  {
    CUDAContextGuard guard {m_context};

    CUdevice device;
    KMM_CUDA_CHECK(cuCtxGetDevice(&device));

    // CUDA assumes maxSize is ignored if its zero, while this constructor uses max_size==MAX
    if (max_size == std::numeric_limits<size_t>::max()) {
        max_size = 0;
    }

    switch (m_kind) {
        case DevicePoolKind::Default:
            KMM_CUDA_CHECK(cuDeviceGetDefaultMemPool(&m_pool, device));
            break;

        case DevicePoolKind::Create:
            CUmemPoolProps props;
            ::memset(&props, 0, sizeof(CUmemPoolProps));

            props.allocType = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            props.handleTypes = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
            props.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            props.maxSize = max_size;
            props.location.id = device;

            KMM_CUDA_CHECK(cuMemPoolCreate(&m_pool, &props));
            break;
    }
}

DevicePoolAllocator::~DevicePoolAllocator() {
    CUDAContextGuard guard {m_context};

    switch (m_kind) {
        case DevicePoolKind::Default:
            // No need to destroy the default pool
            break;
        case DevicePoolKind::Create:
            KMM_CUDA_CHECK(cuMemPoolDestroy(m_pool));
            break;
    }
}

AllocResult DevicePoolAllocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out
) {
    CUdeviceptr device_ptr;
    CUresult result;

    CUDAContextGuard guard {m_context};
    result = cuMemAllocFromPoolAsync(&device_ptr, layout.size_in_bytes, m_pool, stream);

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_CUDA_CHECK(result);
    *addr_out = (void*)device_ptr;
    spdlog::trace("allocate {} bytes of device memory on stream {} (addr: {})", layout.size_in_bytes, stream.id(), *addr_out);
    return AllocResult::Success;
}

void DevicePoolAllocator::deallocate_async(
    const DeviceStream& stream,
    void* addr,
    BufferLayout layout
) {
    CUdeviceptr device_ptr = (CUdeviceptr)addr;
    spdlog::trace("deallocate {} bytes of device memory on stream {} (addr: {})", layout.size_in_bytes, stream.id(), addr);

    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFreeAsync(device_ptr, stream));
}


AllocResult DevicePoolAllocator::allocate(BufferLayout layout, void** addr_out) {
    CUdeviceptr device_ptr;
    CUresult result;

    CUDAContextGuard guard {m_context};

    // Route through the pool (via the legacy default stream) rather than `cuMemAlloc`, so
    // this allocation is still subject to the pool's `maxSize` and gets reclaimed by `trim`.
    result = cuMemAllocFromPoolAsync(&device_ptr, layout.size_in_bytes, m_pool, nullptr);

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_CUDA_CHECK(result);
    KMM_CUDA_CHECK(cuStreamSynchronize(nullptr));

    spdlog::trace("allocate {} bytes of device memory on stream NULL (addr: {})", layout.size_in_bytes, *addr_out);
    *addr_out = (void*)device_ptr;
    return AllocResult::Success;
}

void DevicePoolAllocator::deallocate(void* addr, BufferLayout layout) {
    CUdeviceptr device_ptr = (CUdeviceptr)addr;
    spdlog::trace("deallocate {} bytes of device memory on stream NULL (addr: {})", layout.size_in_bytes, addr);

    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFreeAsync(device_ptr, nullptr));
    KMM_CUDA_CHECK(cuStreamSynchronize(nullptr));
}

void DevicePoolAllocator::trim(size_t nbytes_remaining) {
    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemPoolTrimTo(m_pool, nbytes_remaining));
}

}  // namespace kmm
