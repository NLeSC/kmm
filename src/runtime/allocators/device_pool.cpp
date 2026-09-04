#include <cstring>
#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/device_pool.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

DevicePoolAllocator::DevicePoolAllocator(
    g_context_t context,
    DevicePoolKind kind,
    size_t max_size
) :
    m_context(context),
    m_pool(nullptr),
    m_kind(kind) {
    GPUContextGuard guard {m_context};

    g_device_t device;
    KMM_GPU_CHECK(g_ctx_get_device(&device));

    // CUDA assumes maxSize is ignored if its zero, while this constructor uses max_size==MAX
    if (max_size == std::numeric_limits<size_t>::max()) {
        max_size = 0;
    }

    switch (m_kind) {
        case DevicePoolKind::Default:
            KMM_GPU_CHECK(g_device_get_default_mem_pool(&m_pool, device));
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

            KMM_GPU_CHECK(g_mem_pool_create(&m_pool, &props));
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
            KMM_GPU_CHECK(g_mem_pool_destroy(m_pool));
            break;
    }
}

AllocResult DevicePoolAllocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out
) {
    g_device_ptr_t device_ptr;
    g_result_t result;

    GPUContextGuard guard {m_context};
    result = g_mem_alloc_from_pool_async(&device_ptr, layout.size_in_bytes, m_pool, stream);

    if (result == G_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    *addr_out = (void*)device_ptr;
    spdlog::trace(
        "allocate {} bytes of device memory on stream {} (addr: {})",
        layout.size_in_bytes,
        stream.id(),
        *addr_out
    );
    return AllocResult::Success;
}

void DevicePoolAllocator::deallocate_async(
    const DeviceStream& stream,
    void* addr,
    BufferLayout layout
) {
    g_device_ptr_t device_ptr = (g_device_ptr_t)addr;
    spdlog::trace(
        "deallocate {} bytes of device memory on stream {} (addr: {})",
        layout.size_in_bytes,
        stream.id(),
        addr
    );

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_mem_free_async(device_ptr, stream));
}

AllocResult DevicePoolAllocator::allocate(BufferLayout layout, void** addr_out) {
    g_device_ptr_t device_ptr;
    g_result_t result;

    GPUContextGuard guard {m_context};

    // Route through the pool (via the legacy default stream) rather than `gpuMemAlloc`, so
    // this allocation is still subject to the pool's `maxSize` and gets reclaimed by `trim`.
    result = g_mem_alloc_from_pool_async(&device_ptr, layout.size_in_bytes, m_pool, nullptr);

    if (result == G_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_GPU_CHECK(result);
    KMM_GPU_CHECK(g_stream_synchronize(nullptr));

    spdlog::trace(
        "allocate {} bytes of device memory on stream NULL (addr: {})",
        layout.size_in_bytes,
        *addr_out
    );
    *addr_out = (void*)device_ptr;
    return AllocResult::Success;
}

void DevicePoolAllocator::deallocate(void* addr, BufferLayout layout) {
    g_device_ptr_t device_ptr = (g_device_ptr_t)addr;
    spdlog::trace(
        "deallocate {} bytes of device memory on stream NULL (addr: {})",
        layout.size_in_bytes,
        addr
    );

    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_mem_free_async(device_ptr, nullptr));
    KMM_GPU_CHECK(g_stream_synchronize(nullptr));
}

void DevicePoolAllocator::trim(size_t nbytes_remaining) {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_mem_pool_trim_to(m_pool, nbytes_remaining));
}

}  // namespace kmm
