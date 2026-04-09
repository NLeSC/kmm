#include <unordered_map>

#include "spdlog/spdlog.h"

#include "kmm/memops/gpu_fill.hpp"
#include "kmm/memops/host_fill.hpp"
#include "kmm/runtime/memory_system.hpp"

namespace kmm {

struct MemorySystemImpl::Device {
    KMM_NOT_COPYABLE(Device)

  public:
    GPUContextHandle context;
    std::unique_ptr<AsyncAllocator> allocator;

    DeviceStream h2d_stream;
    DeviceStream d2h_stream;
    DeviceStream h2d_hi_stream;  // high priority stream
    DeviceStream d2h_hi_stream;  // high priority stream

    Device(
        GPUContextHandle context,
        std::unique_ptr<AsyncAllocator> allocator,
        DeviceStreamManager& streams
    ) :
        context(context),
        allocator(std::move(allocator)),
        h2d_stream(streams.create_stream(context, false)),
        d2h_stream(streams.create_stream(context, false)),
        h2d_hi_stream(streams.create_stream(context, true)),
        d2h_hi_stream(streams.create_stream(context, true)) {}
};

MemorySystemImpl::MemorySystemImpl(
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::vector<GPUContextHandle> device_contexts,
    std::unique_ptr<AsyncAllocator> host_mem,
    std::vector<std::unique_ptr<AsyncAllocator>> device_mems
) :
    m_streams(stream_manager),
    m_host(std::move(host_mem))

{
    KMM_ASSERT(device_contexts.size() == device_mems.size());
    KMM_ASSERT(device_contexts.size() <= MAX_DEVICES);

    for (size_t i = 0; i < device_contexts.size(); i++) {
        m_devices[i] = std::make_unique<Device>(
            device_contexts[i],
            std::move(device_mems[i]),
            *stream_manager
        );
    }
}

MemorySystemImpl::~MemorySystemImpl() {}

void MemorySystemImpl::make_progress() {
    m_host->make_progress();

    for (const auto& device : m_devices) {
        if (device == nullptr) {
            break;
        }

        device->allocator->make_progress();
    }
}

void MemorySystemImpl::trim_host(size_t bytes_remaining) {
    m_host->trim(bytes_remaining);
}

AllocationResult MemorySystemImpl::allocate_host(
    size_t nbytes,
    DeviceId device_affinity,
    void** ptr_out,
    DeviceEventSet& deps_out
) {
    // TODO: take into account device_affinity

    auto result = m_host->allocate_async(nbytes, ptr_out, deps_out);
    if (result != AllocationResult::Success) {
        return result;
    }

    deps_out.remove_ready(*m_streams);
    return AllocationResult::Success;
}

void MemorySystemImpl::deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps) {
    deps.remove_ready(*m_streams);
    m_host->deallocate_async(ptr, nbytes, std::move(deps));
}

void MemorySystemImpl::trim_device(size_t bytes_remaining) {
    for (const auto& device : m_devices) {
        if (device != nullptr) {
            device->allocator->trim(bytes_remaining);
        }
    }
}

AllocationResult MemorySystemImpl::allocate_device(
    DeviceId device_id,
    size_t nbytes,
    g_device_ptr_t* ptr_out,
    DeviceEventSet& deps_out
) {
    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];
    void* addr;

    GPUContextGuard guard {device.context};

    auto result = device.allocator->allocate_async(nbytes, &addr, deps_out);
    if (result != AllocationResult::Success) {
        return result;
    }

    deps_out.remove_ready(*m_streams);
    *ptr_out = (g_device_ptr_t)addr;
    return AllocationResult::Success;
}

void MemorySystemImpl::deallocate_device(
    DeviceId device_id,
    g_device_ptr_t ptr,
    size_t nbytes,
    DeviceEventSet deps
) {
    deps.remove_ready(*m_streams);

    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];

    GPUContextGuard guard {device.context};
    device.allocator->deallocate_async((void*)ptr, nbytes, std::move(deps));
}

// Copies smaller than this threshold are put onto a high priority stream. This can improve
// performance since small copy jobs (like copying a single number) are prioritized over large
// slow copy jobs of several gigabytes.
static constexpr size_t HIGH_PRIORITY_THRESHOLD = 1024L * 1024;

DeviceEvent MemorySystemImpl::copy_host_to_device(
    DeviceId device_id,
    const void* src_addr,
    g_device_ptr_t dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.h2d_hi_stream : device.h2d_stream;

    GPUContextGuard guard {device.context};
    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(g_memcpy_h_to_d_async(dst_addr, src_addr, nbytes, stream));
    });
}

DeviceEvent MemorySystemImpl::copy_device_to_host(
    DeviceId device_id,
    g_device_ptr_t src_addr,
    void* dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.d2h_hi_stream : device.d2h_stream;

    GPUContextGuard guard {device.context};
    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(g_memcpy_d_to_h_async(dst_addr, src_addr, nbytes, stream));
    });
}

DeviceEvent MemorySystemImpl::copy_device_to_device(
    DeviceId src_device_id,
    DeviceId dst_device_id,
    g_device_ptr_t src_addr,
    g_device_ptr_t dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    KMM_ASSERT(m_devices[dst_device_id] && m_devices[src_device_id]);
    auto& src_device = *m_devices[src_device_id];
    auto& dst_device = *m_devices[dst_device_id];
    auto stream =
        nbytes <= HIGH_PRIORITY_THRESHOLD ? dst_device.h2d_hi_stream : dst_device.h2d_stream;

    GPUContextGuard guard {dst_device.context};
    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(g_memcpy_peer_async(
            dst_addr,
            dst_device.context,
            dst_device_id,
            src_addr,
            src_device.context,
            src_device_id,
            nbytes,
            stream
        ));
    });
}

}  // namespace kmm