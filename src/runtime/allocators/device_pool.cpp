#include <cstring>
#include <utility>

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
    m_kind(kind),
    m_bytes_in_use(0),
    m_bytes_limit(max_size) {
    CUDAContextGuard guard {m_context};

    CUdevice device;
    KMM_CUDA_CHECK(cuCtxGetDevice(&device));

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
            props.location.id = device;

            KMM_CUDA_CHECK(cuMemPoolCreate(&m_pool, &props));
            break;
    }
}

DevicePoolAllocator::~DevicePoolAllocator() {
    // wait until all evenst are done
    while (!m_pending_deallocs.empty()) {
        auto front = m_pending_deallocs.front();
        front.event.synchronize();
        m_bytes_in_use -= front.nbytes;
        m_pending_deallocs.pop_front();
    }

    KMM_ASSERT(m_bytes_in_use == 0);

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
    void** addr_out,
    DeviceEventSet& deps_out
) {
    size_t nbytes = layout.size_in_bytes;
    poll();

    if (!ensure_enough_space(stream, nbytes)) {
        return AllocResult::ErrorOutOfMemory;
    }

    CUdeviceptr device_ptr;
    CUresult result;

    {
        CUDAContextGuard guard {m_context};
        result = cuMemAllocFromPoolAsync(&device_ptr, nbytes, m_pool, stream.get());
    }

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return AllocResult::ErrorOutOfMemory;
    }

    KMM_CUDA_CHECK(result);

    auto event = stream.record_event();
    m_bytes_in_use += nbytes;
    deps_out.insert(event);
    *addr_out = (void*)device_ptr;
    return AllocResult::Success;
}

void DevicePoolAllocator::deallocate_async(
    const DeviceStream& stream,
    void* addr,
    BufferLayout layout,
    DeviceEventSet deps
) {
    CUdeviceptr device_ptr = (CUdeviceptr)addr;

    auto event = stream.with_stream(deps, [&](auto stream) {
        CUDAContextGuard guard {m_context};
        KMM_CUDA_CHECK(cuMemFreeAsync(device_ptr, stream));
    });

    m_pending_deallocs.push_back({addr, layout.size_in_bytes, event});
}

void DevicePoolAllocator::poll() {
    while (!m_pending_deallocs.empty()) {
        auto front = m_pending_deallocs.front();

        if (!front.event.is_ready()) {
            break;
        }

        m_bytes_in_use -= front.nbytes;
        m_pending_deallocs.pop_front();
    }
}

void DevicePoolAllocator::trim(size_t nbytes_remaining) {
    while (m_bytes_in_use > nbytes_remaining) {
        if (m_pending_deallocs.empty()) {
            break;
        }

        auto& d = m_pending_deallocs.front();
        d.event.synchronize();

        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }

    CUDAContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemPoolTrimTo(m_pool, nbytes_remaining));
}

bool DevicePoolAllocator::is_allocation_allowed(size_t nbytes) const {
    return m_bytes_limit - m_bytes_in_use >= nbytes;
}

bool DevicePoolAllocator::ensure_enough_space(const DeviceStream& stream, size_t nbytes) {
    if (is_allocation_allowed(nbytes)) {
        return true;
    }

    auto it = m_pending_deallocs.begin();

    // first, we try to find pending deallocations where the dependencies naturally are preceding
    // the given stream. There are already deallocated when the stream reaches this point.
    while (it != m_pending_deallocs.end()) {
        if (stream.preceded_by(it->event)) {
            stream.wait_on_event(it->event);
            m_bytes_in_use -= it->nbytes;
            it = m_pending_deallocs.erase(it);

            if (is_allocation_allowed(nbytes)) {
                return true;
            }
        } else {
            it++;
        }
    }

    it = m_pending_deallocs.begin();

    // second, we forcefully wait for pending deallocations. The stream must wait until enough
    // memory is available for the new allocations.
    while (it != m_pending_deallocs.end()) {
        stream.wait_on_events(it->event);
        m_bytes_in_use -= it->nbytes;
        it = m_pending_deallocs.erase(it);

        if (is_allocation_allowed(nbytes)) {
            return true;
        }
    }

    return false;
}

}  // namespace kmm
