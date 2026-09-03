#include <algorithm>
#include <utility>

#include "spdlog/spdlog.h"

#include "kmm/core/checked_math.hpp"
#include "kmm/runtime/allocators/arena.hpp"
#include "kmm/runtime/allocators/device.hpp"
#include "kmm/runtime/allocators/device_pool.hpp"
#include "kmm/runtime/allocators/limit.hpp"
#include "kmm/runtime/allocators/managed.hpp"
#include "kmm/runtime/allocators/pinned.hpp"
#include "kmm/runtime/device_data_streams.hpp"
#include "kmm/runtime/memops/fill_gpu.hpp"
#include "kmm/runtime/memops/reduction_gpu.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

struct MemorySystem::DeviceState {
    DeviceState(g_context_t context, g_device_t ordinal, std::unique_ptr<Allocator> allocator) :
        context(context),
        ordinal(ordinal),
        allocator(std::move(allocator)) {}

    g_context_t context;
    g_device_t ordinal;
    std::unique_ptr<Allocator> allocator;

    MemoryStats stats;

    void record_allocation(size_t nbytes) {
        stats.record_allocation(nbytes);
    }

    void record_deallocation(size_t nbytes) {
        stats.record_deallocation(nbytes);
    }
};

static std::unique_ptr<Allocator> make_host_allocator(
    const RuntimeConfig& config,
    DeviceEventRegistry events,
    g_context_t context
) {
    std::unique_ptr<Allocator> allocator;
    allocator = std::make_unique<PinnedMemoryAllocator>(context);

    if (config.host_memory_limit != std::numeric_limits<size_t>::max()) {
        allocator = std::make_unique<LimitAllocator>(
            std::move(allocator),
            events,
            config.host_memory_limit
        );
    }

    if (config.host_memory_kind != HostMemoryKind::NoPool && config.host_memory_block_size > 0) {
        allocator =
            std::make_unique<ArenaAllocator>(std::move(allocator), config.host_memory_block_size);
    }

    return allocator;
}

static std::unique_ptr<Allocator> make_device_allocator(
    const RuntimeConfig& config,
    DeviceEventRegistry events,
    g_context_t context,
    size_t device_memory_size
) {
    std::unique_ptr<Allocator> allocator;

    size_t limit = config.device_memory_limit;

    if (config.device_memory_keep_free > 0) {
        if (device_memory_size <= config.device_memory_keep_free) {
            throw std::runtime_error(
                fmt::format(
                    "cannot reserve {} bytes on GPU, only {} bytes are available",
                    config.device_memory_keep_free,
                    device_memory_size
                )
            );
        }

        limit = std::min(limit, device_memory_size - config.device_memory_keep_free);
    }

    switch (config.device_memory_kind) {
        case DeviceMemoryKind::DefaultPool:
            allocator =
                std::make_unique<DevicePoolAllocator>(context, DevicePoolKind::Default, limit);
            break;
        case DeviceMemoryKind::PrivatePool:
            allocator =
                std::make_unique<DevicePoolAllocator>(context, DevicePoolKind::Create, limit);
            break;
        default:
            allocator = std::make_unique<DeviceMemoryAllocator>(context);
            break;
    }

    if (limit != std::numeric_limits<size_t>::max()) {
        allocator = std::make_unique<LimitAllocator>(std::move(allocator), events, limit);
    }

    if (config.device_memory_kind != DeviceMemoryKind::NoPool
        && config.device_memory_block_size > 0) {
        allocator =
            std::make_unique<ArenaAllocator>(std::move(allocator), config.device_memory_block_size);
    }

    return allocator;
}

MemorySystem::MemorySystem(
    const SystemInfo& system_info,
    DeviceEventRegistry events,
    const RuntimeConfig& config
) :
    m_events(events),
    m_streams(
        system_info,
        events,
        config.device_concurrent_streams,
        config.device_concurrent_streams,
        config.device_concurrent_streams
    ),
    m_num_devices(system_info.num_devices()) {
    spdlog::info("initializing memory system with {} device(s)", m_num_devices);

    g_context_t host_context =
        m_num_devices > 0 ? system_info.device(DeviceId(0)).context() : nullptr;
    m_host_allocator = make_host_allocator(config, events, host_context);

    if (m_num_devices > 0) {
        m_managed_allocator = std::make_unique<ManagedMemoryAllocator>(host_context);
    }

    for (size_t i = 0; i < m_num_devices; i++) {
        const auto& info = system_info.device(DeviceId(i));

        m_devices[i] = std::make_unique<DeviceState>(
            info.context(),
            info.device_ordinal(),
            make_device_allocator(config, events, info.context(), info.total_memory_size())
        );
    }

    // Determine, and where possible enable, peer-to-peer access between every device pair.
    for (size_t i = 0; i < m_num_devices; i++) {
        m_peer_access[i][i] = true;

        for (size_t j = i + 1; j < m_num_devices; j++) {
            int i_can_access_j = 0;
            int j_can_access_i = 0;

            KMM_GPU_CHECK(g_device_can_access_peer(
                &i_can_access_j,
                system_info.device(DeviceId(i)).device_ordinal(),
                system_info.device(DeviceId(j)).device_ordinal()
            ));

            KMM_GPU_CHECK(g_device_can_access_peer(
                &j_can_access_i,
                system_info.device(DeviceId(j)).device_ordinal(),
                system_info.device(DeviceId(i)).device_ordinal()
            ));

            m_peer_access[i][j] = i_can_access_j != 0;
            m_peer_access[j][i] = j_can_access_i != 0;

            if (i_can_access_j) {
                GPUContextGuard guard {m_devices[i]->context};
                g_result_t result = g_ctx_enable_peer_access(m_devices[j]->context, 0);

                if (result != GPU_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                    KMM_GPU_CHECK(result);
                }
            }

            if (j_can_access_i) {
                GPUContextGuard guard {m_devices[j]->context};
                g_result_t result = g_ctx_enable_peer_access(m_devices[i]->context, 0);

                if (result != GPU_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                    KMM_GPU_CHECK(result);
                }
            }

            spdlog::debug(
                "peer access between device {} and device {}: {} -> {} = {}, {} -> {} = {}",
                i,
                j,
                i,
                j,
                i_can_access_j != 0,
                j,
                i,
                j_can_access_i != 0
            );
        }
    }
}

MemorySystem::~MemorySystem() {
    spdlog::info("memory system stats:");
    spdlog::info(" - host:");
    spdlog::info("   - allocated: {} bytes", m_host_stats.bytes_allocated);
    spdlog::info("   - allocated at peak: {} bytes", m_host_stats.max_bytes_inuse);

    for (size_t i = 0; i < m_num_devices; i++) {
        spdlog::info("   - copied to device {}: {} bytes", i, m_host_stats.bytes_to_device[i]);
        spdlog::info("   - copied from device {}: {} bytes", i, m_devices[i]->stats.bytes_to_host);
    }

    if (const auto* arena = dynamic_cast<const ArenaAllocator*>(m_host_allocator.get())) {
        spdlog::info("   - reserved from underlying allocator: {} bytes", arena->bytes_reserved());
    }

    for (size_t i = 0; i < m_num_devices; i++) {
        const auto& state = *m_devices[i];

        spdlog::info(" - device {}:", i);
        spdlog::info("   - allocated: {} bytes", state.stats.bytes_allocated);
        spdlog::info("   - allocated at peak: {} bytes", state.stats.max_bytes_inuse);
        spdlog::info("   - copied to host: {} bytes", state.stats.bytes_to_host);
        spdlog::info("   - copied from host: {} bytes", m_host_stats.bytes_to_device[i]);

        for (size_t j = 0; j < m_num_devices; j++) {
            if (j != i) {
                spdlog::info(
                    "   - copied to device {}: {} bytes",
                    j,
                    state.stats.bytes_to_device[j]
                );
                spdlog::info(
                    "   - copied from device {}: {} bytes",
                    j,
                    m_devices[j]->stats.bytes_to_device[i]
                );
            }
        }

        if (const auto* arena = dynamic_cast<const ArenaAllocator*>(state.allocator.get())) {
            spdlog::info(
                "   - reserved from underlying allocator: {} bytes",
                arena->bytes_reserved()
            );
        }
    }
}

MemorySystem::DeviceState& MemorySystem::device_state(DeviceId id) const {
    KMM_ASSERT(id.get() < m_num_devices);
    return *m_devices[id.get()];
}

void MemorySystem::make_progress() {
    m_host_allocator->poll();

    for (size_t i = 0; i < m_num_devices; i++) {
        m_devices[i]->allocator->poll();
    }

    m_streams.make_progress();
}

void MemorySystem::trim_host(size_t bytes_remaining) {
    m_host_allocator->trim(bytes_remaining);
}

void MemorySystem::trim_device(size_t bytes_remaining) {
    for (size_t i = 0; i < m_num_devices; i++) {
        m_devices[i]->allocator->trim(bytes_remaining);
    }
}

template<typename F>
DeviceEvent schedule_onto_stream(
    bool use_hint,
    const DeviceEventRegistry& events,
    DeviceDataStreams& streams,
    DeviceId device_id,
    StreamKind kind,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in,
    F callback
) {
    if (stream_hint.is_null() || !use_hint) {
        return streams.submit(device_id, kind, deps_in, [&](auto stream_id) -> uint64_t {
            return callback(DeviceStream {events, stream_id});
        });
    } else {
        return events.submit(stream_hint, deps_in, [&](auto stream) {
            callback(DeviceStream {events, stream_hint});
        });
    }
}

DeviceId MemorySystem::affinity_for_stream(const DeviceStreamId& stream_hint) {
    if (!stream_hint.is_null()) {
        for (size_t i = 0; i < m_num_devices; i++) {
            if (m_events.has_context(stream_hint, m_devices[i]->context)) {
                return DeviceId(i);
            }
        }
    }

    return DeviceId(0);
}

AllocResult MemorySystem::allocate_host(
    BufferLayout layout,
    void** ptr_out,
    const DeviceStreamId& stream_hint,
    DeviceEventSet& deps_out
) {
    AllocResult result;

    if (stream_hint.is_null()) {
        result = m_host_allocator->allocate(layout, ptr_out);
    } else {
        result =
            m_host_allocator->allocate_async(DeviceStream {m_events, stream_hint}, layout, ptr_out);

        deps_out.insert(m_events.record(stream_hint));
    }

    if (result == AllocResult::Success) {
        m_host_stats.record_allocation(layout.size_in_bytes);
    }

    spdlog::trace(
        "allocate {} bytes of host memory (addr: {}, result: {})",
        layout.size_in_bytes,
        *ptr_out,
        static_cast<int>(result)
    );

    return result;
}

void MemorySystem::deallocate_host(
    void* ptr,
    BufferLayout layout,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace("deallocate {} bytes of host memory (addr: {})", layout.size_in_bytes, ptr);

    m_host_stats.record_deallocation(layout.size_in_bytes);

    if (stream_hint.is_null()) {
        m_host_allocator->deallocate(ptr, layout);
    } else {
        m_events.wait_on_event(stream_hint, deps_in);
        m_host_allocator->deallocate_async(DeviceStream {m_events, stream_hint}, ptr, layout);
    }
}

AllocResult MemorySystem::allocate_managed(
    BufferLayout layout,
    void** ptr_out,
    const DeviceStreamId& stream_hint,
    DeviceEventSet& deps_out
) {
    if (!m_managed_allocator) {
        return AllocResult::ErrorUnsupported;
    }

    AllocResult result;

    if (stream_hint.is_null()) {
        result = m_managed_allocator->allocate(layout, ptr_out);
    } else {
        result = m_managed_allocator->allocate_async(
            DeviceStream {m_events, stream_hint},
            layout,
            ptr_out
        );

        deps_out.insert(m_events.record(stream_hint));
    }

    if (result == AllocResult::Success) {
        m_managed_stats.record_allocation(layout.size_in_bytes);
    }

    spdlog::trace(
        "allocate {} bytes of managed memory (addr: {}, result: {})",
        layout.size_in_bytes,
        *ptr_out,
        static_cast<int>(result)
    );

    return result;
}

void MemorySystem::deallocate_managed(
    void* ptr,
    BufferLayout layout,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace("deallocate {} bytes of managed memory (addr: {})", layout.size_in_bytes, ptr);

    m_managed_stats.record_deallocation(layout.size_in_bytes);

    if (stream_hint.is_null()) {
        m_managed_allocator->deallocate(ptr, layout);
    } else {
        m_events.wait_on_event(stream_hint, deps_in);
        m_managed_allocator->deallocate_async(DeviceStream {m_events, stream_hint}, ptr, layout);
    }
}

void MemorySystem::prefetch_managed(
    MemoryId memory_id,
    void* ptr,
    BufferLayout layout,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    if (stream_hint.is_null() || !memory_id.is_device()) {
        return;
    }

    auto device_id = memory_id.as_device();
    if (!same_context(device_id, stream_hint)) {
        return;
    }

    spdlog::trace(
        "prefetch {} bytes of managed memory to {} (addr: {})",
        layout.size_in_bytes,
        memory_id,
        ptr
    );

    GPUContextGuard guard {device_state(device_id).context};
    m_events.wait_on_event(stream_hint, deps);

    KMM_GPU_CHECK(g_mem_prefetch_async(
        reinterpret_cast<g_device_ptr_t>(ptr),
        layout.size_in_bytes,
        device_state(device_id).ordinal,
        m_events.get(stream_hint)
    ));
}

void* MemorySystem::translate_host_pointer(DeviceId device_id, void* host_ptr) const {
    GPUContextGuard guard {device_state(device_id).context};

    g_device_ptr_t dptr;
    KMM_GPU_CHECK(g_mem_host_get_device_pointer(&dptr, host_ptr, 0));
    return reinterpret_cast<void*>(dptr);
}

bool MemorySystem::same_context(kmm::DeviceId device_id, const kmm::DeviceStreamId& stream_hint) {
    return m_events.has_context(stream_hint, device_state(device_id).context);
}

AllocResult MemorySystem::allocate_device(
    DeviceId device_id,
    BufferLayout layout,
    g_device_ptr_t* ptr_out,
    const DeviceStreamId& stream_hint,
    DeviceEventSet& deps_out
) {
    void* addr = nullptr;
    AllocResult result;

    auto event = schedule_onto_stream(
        same_context(device_id, stream_hint),
        m_events,
        m_streams,
        device_id,
        StreamKind::DeviceToDevice,
        stream_hint,
        {},
        [&](const auto& stream) {
            result = device_state(device_id).allocator->allocate_async(stream, layout, &addr);
            return 0;
        }
    );

    deps_out.insert(event);
    *ptr_out = reinterpret_cast<g_device_ptr_t>(addr);

    if (result == AllocResult::Success) {
        device_state(device_id).record_allocation(layout.size_in_bytes);
    }

    spdlog::trace(
        "allocate {} bytes of device memory on device {} (addr: {:#x}, result: {})",
        layout.size_in_bytes,
        device_id,
        *ptr_out,
        static_cast<int>(result)
    );

    return result;
}

void MemorySystem::deallocate_device(
    DeviceId device_id,
    g_device_ptr_t ptr,
    BufferLayout layout,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace(
        "deallocate {} bytes of device memory on device {} (addr: {:#x})",
        layout.size_in_bytes,
        device_id,
        ptr
    );

    schedule_onto_stream(
        same_context(device_id, stream_hint),
        m_events,
        m_streams,
        device_id,
        StreamKind::DeviceToDevice,
        stream_hint,
        deps_in,
        [&](const auto& stream) {
            device_state(device_id).allocator->deallocate_async(
                stream,
                reinterpret_cast<void*>(ptr),
                layout
            );

            return 0;
        }
    );

    device_state(device_id).record_deallocation(layout.size_in_bytes);
}

DeviceEvent MemorySystem::copy_host_to_device(
    DeviceId device_id,
    const void* src_addr,
    g_device_ptr_t dst_addr,
    size_t nbytes,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace(
        "copy {} bytes from host (addr: {}) to device {} (addr: {:#x})",
        nbytes,
        src_addr,
        device_id,
        dst_addr
    );

    bool use_hint =
        same_context(device_id, stream_hint) && m_events.is_latest_in(stream_hint, deps_in);

    auto event = schedule_onto_stream(
        use_hint,
        m_events,
        m_streams,
        device_id,
        StreamKind::HostToDevice,
        stream_hint,
        deps_in,
        [&](g_stream_t stream) {
            KMM_GPU_CHECK(g_memcpy_h_to_d_async(dst_addr, src_addr, nbytes, (g_stream_t)stream));
            return nbytes;
        }
    );

    m_host_stats.bytes_to_device[device_id.get()] += nbytes;
    return event;
}

DeviceEvent MemorySystem::copy_device_to_host(
    DeviceId device_id,
    g_device_ptr_t src_addr,
    void* dst_addr,
    size_t nbytes,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace(
        "copy {} bytes from device {} (addr: {:#x}) to host (addr: {})",
        nbytes,
        device_id,
        src_addr,
        dst_addr
    );

    bool use_hint =
        same_context(device_id, stream_hint) && m_events.is_latest_in(stream_hint, deps_in);

    auto event = schedule_onto_stream(
        use_hint,
        m_events,
        m_streams,
        device_id,
        StreamKind::DeviceToHost,
        stream_hint,
        deps_in,
        [&](g_stream_t stream) {
            KMM_GPU_CHECK(g_memcpy_d_to_h_async(dst_addr, src_addr, nbytes, (g_stream_t)stream));
            return nbytes;
        }
    );

    device_state(device_id).stats.bytes_to_host += nbytes;
    return event;
}

DeviceEvent MemorySystem::copy_device_to_device(
    DeviceId src_device,
    DeviceId dst_device,
    g_device_ptr_t src_addr,
    g_device_ptr_t dst_addr,
    size_t nbytes,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace(
        "copy {} bytes from device {} (addr: {:#x}) to device {} (addr: {:#x})",
        nbytes,
        src_device,
        src_addr,
        dst_device,
        dst_addr
    );

    auto event = m_streams.submit(  //
        dst_device,
        StreamKind::DeviceToDevice,
        deps_in,
        [&](auto stream_id) -> uint64_t {
            KMM_GPU_CHECK(g_memcpy_peer_async(
                dst_addr,
                device_state(dst_device).context,
                device_state(dst_device).ordinal,
                src_addr,
                device_state(src_device).context,
                device_state(src_device).ordinal,
                nbytes,
                m_events.get(stream_id)
            ));

            return nbytes;
        }
    );

    device_state(src_device).stats.bytes_to_device[dst_device.get()] += nbytes;
    return event;
}

AllocResult MemorySystem::allocate_host_and_copy_from_device(
    BufferLayout layout,
    void** dst_addr,
    DeviceId device_id,
    g_device_ptr_t src_addr,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in,
    DeviceEvent& dep_out
) {
    AllocResult result;
    bool use_hint =
        same_context(device_id, stream_hint) && m_events.is_latest_in(stream_hint, deps_in);

    dep_out = schedule_onto_stream(
        use_hint,
        m_events,
        m_streams,
        device_id,
        StreamKind::DeviceToHost,
        stream_hint,
        deps_in,
        [&](const auto& stream) {
            void* addr = nullptr;

            result = m_host_allocator->allocate_async(stream, layout, &addr);

            if (result != AllocResult::Success) {
                return size_t(0);
            }

            *dst_addr = addr;

            try {
                KMM_GPU_CHECK(g_memcpy_d_to_h_async(
                    *dst_addr,
                    src_addr,
                    layout.size_in_bytes,
                    (g_stream_t)stream
                ));
                return layout.size_in_bytes;
            } catch (...) {
                m_host_allocator->deallocate_async(stream, addr, layout);
                throw;
            }
        }
    );

    if (result == AllocResult::Success) {
        device_state(device_id).stats.bytes_to_host += layout.size_in_bytes;
    }

    return result;
}

AllocResult MemorySystem::allocate_device_and_copy_from_host(
    DeviceId device_id,
    BufferLayout layout,
    g_device_ptr_t* dst_addr,
    const void* src_addr,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in,
    DeviceEvent& dep_out
) {
    AllocResult result;
    bool use_hint =
        same_context(device_id, stream_hint) && m_events.is_latest_in(stream_hint, deps_in);

    dep_out = schedule_onto_stream(
        use_hint,
        m_events,
        m_streams,
        device_id,
        StreamKind::HostToDevice,
        stream_hint,
        deps_in,
        [&](const auto& stream) {
            void* addr = nullptr;

            result = device_state(device_id).allocator->allocate_async(stream, layout, &addr);

            if (result != AllocResult::Success) {
                return size_t(0);
            }

            *dst_addr = reinterpret_cast<g_device_ptr_t>(addr);

            try {
                spdlog::trace(
                    "allocate {} bytes on device {} and copy from host (addr: {}, dst addr: {:#x})",
                    layout.size_in_bytes,
                    device_id,
                    src_addr,
                    *dst_addr
                );
                KMM_GPU_CHECK(g_memcpy_h_to_d_async(
                    *dst_addr,
                    src_addr,
                    layout.size_in_bytes,
                    (g_stream_t)stream
                ));
                return layout.size_in_bytes;
            } catch (...) {
                device_state(device_id).allocator->deallocate(addr, layout);
                throw;
            }
        }
    );

    if (result == AllocResult::Success) {
        device_state(device_id).record_allocation(layout.size_in_bytes);
        m_host_stats.bytes_to_device[device_id.get()] += layout.size_in_bytes;
    }

    return result;
}

DeviceEvent MemorySystem::fill_device(
    DeviceId device_id,
    g_device_ptr_t addr,
    const FillDescription& description,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace("fill device {} memory (addr: {:#x})", device_id, addr);

    return schedule_onto_stream(
        same_context(device_id, stream_hint),
        m_events,
        m_streams,
        device_id,
        StreamKind::DeviceToDevice,
        stream_hint,
        deps_in,
        [&](g_stream_t stream) -> size_t {
#if defined(KMM_USE_CUDA) || defined(KMM_USE_HIP)
            memops::fill_gpu(stream, reinterpret_cast<void*>(addr), description);
            return checked_mul<size_t>(description.num_elements(), description.value.length);
#else
            throw std::runtime_error("unsupported operation");
#endif
        }
    );
}

std::future<void> MemorySystem::fill_host(
    void* addr,
    const FillDescription& description,
    const DeviceEventSet& deps_in
) {
    spdlog::trace("fill host memory (addr: {})", addr);

    return std::async(std::launch::async, [this, addr, description, deps_in] {
        m_events.synchronize(deps_in);
        memops::fill(addr, description);
    });
}

DeviceEvent MemorySystem::reduce_device(
    DeviceId device_id,
    g_device_ptr_t src_addr,
    g_device_ptr_t dst_addr,
    g_device_ptr_t scratch_addr,
    const ReductionDescription& description,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in
) {
    spdlog::trace(
        "reduce device {} memory (src addr: {:#x}, dst addr: {:#x})",
        device_id,
        src_addr,
        dst_addr
    );

    return schedule_onto_stream(
        same_context(device_id, stream_hint),
        m_events,
        m_streams,
        device_id,
        StreamKind::DeviceToDevice,
        stream_hint,
        deps_in,
        [&](g_stream_t stream) -> memops_extent_type {
#if defined(KMM_USE_CUDA) || defined(KMM_USE_HIP)
            memops::reduce_gpu(
                stream,
                reinterpret_cast<void*>(src_addr),
                reinterpret_cast<void*>(dst_addr),
                reinterpret_cast<void*>(scratch_addr),
                description
            );
            return description.num_outputs();
#else
            throw std::runtime_error("unsupported operation");
#endif
        }
    );
}

bool MemorySystem::is_copy_supported(MemoryId src, MemoryId dst) const noexcept {
    if (src.is_host() || dst.is_host()) {
        return true;
    }

    auto src_id = src.as_device();
    auto dst_id = dst.as_device();

    if (src_id == dst_id) {
        return true;
    }

    return m_peer_access[src_id.get()][dst_id.get()];
}

}  // namespace kmm
