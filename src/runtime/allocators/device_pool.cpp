#include <cstring>
#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/device_pool.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

DevicePoolAllocator::DevicePoolAllocator(
    GPUContext context,
    DevicePoolKind kind,
    size_t max_size
) :
    m_context(context),
    m_pool(nullptr),
    m_kind(kind)  {
    GPUContextGuard guard {m_context};

    GPUDevice device;
    KMM_GPU_CHECK(gpuCtxGetDevice(&device));

    // CUDA assumes maxSize is ignored if its zero, while this constructor uses max_size==MAX
    if (max_size == std::numeric_limits<size_t>::max()) {
        max_size = 0;
    }

    switch (m_kind) {
        case DevicePoolKind::Default:
            KMM_GPU_CHECK(gpuDeviceGetDefaultMemPool(&m_pool, device));
            break;

        case DevicePoolKind::Create:
#if defined(KMM_USE_CUDA)
            CUmemPoolProps props;
            ::memset(&props, 0, sizeof(CUmemPoolProps));

            props.allocType = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            props.handleTypes = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
            props.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            props.maxSize = max_size;
            props.location.id = device;

            KMM_GPU_CHECK(gpuMemPoolCreate(&m_pool, &props));
#else
            throw std::runtime_error("memory pool is only supported with CUDA backend");
#endif
            break;
    }
}

DevicePoolAllocator::~DevicePoolAllocator() {
    GPUContextGuard guard {m_context};

    switch (m_kind) {
        case DevicePoolKind::Default:
            // No need to destroy the default pool
            break;
        case DevicePoolKind::Create:
            KMM_GPU_CHECK(gpuMemPoolDestroy(m_pool));
            break;
    }
}

AllocResult DevicePoolAllocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out
) {
    GPUDeviceptr device_ptr;
    GPUResult result;

    GPUContextGuard guard {m_context};
    result = gpuMemAllocFromPoolAsync(&device_ptr, layout.size_in_bytes, m_pool, stream);

    if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    *addr_out = (void*)device_ptr;
    spdlog::trace("allocate {} bytes of device memory on stream {} (addr: {})", layout.size_in_bytes, stream.id(), *addr_out);
    return AllocResult::Success;
}

void DevicePoolAllocator::deallocate_async(
    const DeviceStream& stream,
    void* addr,
    BufferLayout layout
) {
    GPUDeviceptr device_ptr = (GPUDeviceptr)addr;
    spdlog::trace("deallocate {} bytes of device memory on stream {} (addr: {})", layout.size_in_bytes, stream.id(), addr);

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemFreeAsync(device_ptr, stream));
}

AllocResult DevicePoolAllocator::allocate(BufferLayout layout, void** addr_out) {
    GPUDeviceptr device_ptr;
    GPUResult result;

    GPUContextGuard guard {m_context};

    // Route through the pool (via the legacy default stream) rather than `gpuMemAlloc`, so
    // this allocation is still subject to the pool's `maxSize` and gets reclaimed by `trim`.
    result = gpuMemAllocFromPoolAsync(&device_ptr, layout.size_in_bytes, m_pool, nullptr);

    if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    KMM_GPU_CHECK(gpuStreamSynchronize(nullptr));

    spdlog::trace("allocate {} bytes of device memory on stream NULL (addr: {})", layout.size_in_bytes, *addr_out);
    *addr_out = (void*)device_ptr;
    return AllocResult::Success;
}

void DevicePoolAllocator::deallocate(void* addr, BufferLayout layout) {
    GPUDeviceptr device_ptr = (GPUDeviceptr)addr;
    spdlog::trace("deallocate {} bytes of device memory on stream NULL (addr: {})", layout.size_in_bytes, addr);

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemFreeAsync(device_ptr, nullptr));
    KMM_GPU_CHECK(gpuStreamSynchronize(nullptr));
}

void DevicePoolAllocator::trim(size_t nbytes_remaining) {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemPoolTrimTo(m_pool, nbytes_remaining));
}

}  // namespace kmm
