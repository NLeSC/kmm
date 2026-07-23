#include "kmm/runtime/allocators/device.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(
    GPUContextHandle context,
    std::shared_ptr<DeviceStreamManager> streams,
    size_t max_bytes
) :
    SyncAllocator(streams, max_bytes),
    m_context(context) {}

AllocationResult PinnedMemoryAllocator::allocate(size_t nbytes, void** addr_out) {
    GPUContextGuard guard {m_context};
    g_result_t result =
        g_mem_host_alloc(addr_out, nbytes, G_MEMHOSTALLOC_PORTABLE | G_MEMHOSTALLOC_DEVICEMAP);

    if (result == G_SUCCESS) {
        return AllocationResult::Success;
    } else if (result == G_ERROR_OUT_OF_MEMORY) {
        return AllocationResult::ErrorOutOfMemory;
    } else {
        throw GPUDriverException("error when calling `cuMemHostAlloc`", result);
    }
}

void PinnedMemoryAllocator::deallocate(void* addr, size_t nbytes) {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_mem_free_host(addr));
}

DeviceMemoryAllocator::DeviceMemoryAllocator(
    GPUContextHandle context,
    std::shared_ptr<DeviceStreamManager> streams,
    size_t max_bytes
) :
    SyncAllocator(streams, max_bytes),
    m_context(context) {}

AllocationResult DeviceMemoryAllocator::allocate(size_t nbytes, void** addr_out) {
    GPUContextGuard guard {m_context};
    g_device_ptr_t ptr;
    g_result_t result = g_mem_alloc(&ptr, nbytes);

    if (result == G_SUCCESS) {
        *addr_out = (void*)ptr;
        return AllocationResult::Success;
    } else if (result == G_ERROR_OUT_OF_MEMORY) {
        return AllocationResult::ErrorOutOfMemory;
    } else {
        throw GPUDriverException("error when calling `g_mem_alloc`", result);
    }
}

void DeviceMemoryAllocator::deallocate(void* addr, size_t nbytes) {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_mem_free((g_device_ptr_t)addr));
}

DevicePoolAllocator::DevicePoolAllocator(
    GPUContextHandle context,
    std::shared_ptr<DeviceStreamManager> streams,
    DevicePoolKind kind,
    size_t max_bytes
) :
    m_context(context),
    m_streams(streams),
    m_alloc_stream(streams->create_stream(context)),
    m_dealloc_stream(streams->create_stream(context)),
    m_kind(kind),
    m_bytes_limit(max_bytes) {
    GPUContextGuard guard {m_context};

    g_device_t device;
    KMM_GPU_CHECK(g_ctx_get_device(&device));

    switch (m_kind) {
        case DevicePoolKind::Default:
            KMM_GPU_CHECK(g_device_get_default_mem_pool(&m_pool, device));
            break;

        case DevicePoolKind::Create:
            g_mem_pool_props_t props;
            ::memset(&props, 0, sizeof(g_mem_pool_props_t));

            props.allocType = G_MEM_ALLOCATION_TYPE_PINNED;
            props.handleTypes = G_MEM_HANDLE_TYPE_NONE;
            props.location.type = G_MEM_LOCATION_TYPE_DEVICE;
            props.location.id = device;

            KMM_GPU_CHECK(g_mem_pool_create(&m_pool, &props));
            break;
    }
}

DevicePoolAllocator::~DevicePoolAllocator() {
    for (auto d : m_pending_deallocs) {
        m_bytes_in_use -= d.nbytes;
        m_streams->wait_until_ready(d.event);
    }

    KMM_ASSERT(m_bytes_in_use == 0);

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

AllocationResult DevicePoolAllocator::allocate_async(
    size_t nbytes,
    void** addr_out,
    DeviceEventSet& deps_out
) {
    make_progress();

    while (m_bytes_limit - m_bytes_in_use < nbytes) {
        if (m_pending_deallocs.empty()) {
            return AllocationResult::ErrorOutOfMemory;
        }

        auto& d = m_pending_deallocs.front();
        m_streams->wait_for_event(m_alloc_stream, d.event);
        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }

    g_device_ptr_t device_ptr;
    g_result_t result = g_result_t(G_ERROR_UNKNOWN);

    auto event = m_streams->with_stream(m_alloc_stream, [&](auto stream) {
        GPUContextGuard guard {m_context};
        result = g_mem_alloc_from_pool_async(&device_ptr, nbytes, m_pool, stream);
    });

    if (result == G_SUCCESS) {
        m_bytes_in_use += nbytes;
        deps_out.insert(event);
        *addr_out = (void*)device_ptr;
        return AllocationResult::Success;
    } else if (result == G_ERROR_OUT_OF_MEMORY) {
        return AllocationResult::ErrorOutOfMemory;
    } else {
        throw GPUDriverException("error while calling `g_mem_alloc_from_pool_async`", result);
    }
}

void DevicePoolAllocator::deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) {
    g_device_ptr_t device_ptr = (g_device_ptr_t)addr;

    auto event = m_streams->with_stream(m_dealloc_stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(g_mem_free_async(device_ptr, stream));
    });

    m_pending_deallocs.push_back({.addr = addr, .nbytes = nbytes, .event = event});
}

void DevicePoolAllocator::make_progress() {
    while (true) {
        if (m_pending_deallocs.empty()) {
            break;
        }

        auto& d = m_pending_deallocs.front();

        if (!m_streams->is_ready(d.event)) {
            break;
        }

        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }
}

void DevicePoolAllocator::trim(size_t nbytes_remaining) {
    while (m_bytes_in_use > nbytes_remaining) {
        if (m_pending_deallocs.empty()) {
            break;
        }

        auto& d = m_pending_deallocs.front();
        m_streams->wait_until_ready(d.event);

        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }

    KMM_GPU_CHECK(g_mem_pool_trim_to(m_pool, nbytes_remaining));
}
}  // namespace kmm