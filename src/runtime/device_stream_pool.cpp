#include "kmm/core/panic.hpp"
#include "kmm/runtime/device_stream_pool.hpp"

namespace kmm {

DeviceStreamPool::DeviceStreamPool(
    const SystemInfo& system_info,
    DeviceStreamRegistry& streams,
    size_t num_data_streams,
    size_t num_compute_streams
) :
    m_num_devices(system_info.num_devices()) {
    KMM_ASSERT(num_data_streams > 0);
    KMM_ASSERT(num_compute_streams > 0);
    KMM_ASSERT(m_num_devices <= MAX_DEVICES);

    for (size_t i = 0; i < m_num_devices; i++) {
        auto context = system_info.device(DeviceId(i)).context();
        auto& dev = m_devices[i];

        dev.data.reserve(num_data_streams);
        for (size_t j = 0; j < num_data_streams; j++) {
            dev.data.emplace_back(streams.create(context));
        }

        dev.compute.reserve(num_compute_streams);
        for (size_t j = 0; j < num_compute_streams; j++) {
            dev.compute.emplace_back(streams.create(context));
        }
    }
}

DeviceStreamPool::~DeviceStreamPool() = default;

DeviceStream DeviceStreamPool::get_stream(
    DeviceId device,
    StreamKind kind,
    const DeviceEventSet& deps
) const {
    KMM_ASSERT(device.get() < m_num_devices);
    auto& dev = m_devices[device.get()];

    auto* entry = (kind == StreamKind::Compute)
        ? pick(dev.compute, dev.next_compute, kind, deps, /* match_direction */ false)
        : pick(dev.data, dev.next_data, kind, deps, /* match_direction */ true);

    entry->last_kind.store(kind, std::memory_order_relaxed);
    entry->stream.wait_on_events(deps);
    return entry->stream;
}

const DeviceStreamPool::PooledStream* DeviceStreamPool::pick(
    const std::vector<PooledStream>& pool,
    std::atomic<size_t>& cursor,
    StreamKind kind,
    const DeviceEventSet& deps,
    bool match_direction
) {
    if (!deps.is_empty()) {
        const PooledStream* same_direction = nullptr;
        const PooledStream* any = nullptr;

        for (const auto& entry : pool) {
            if (!entry.stream.is_latest_in(deps)) {
                continue;
            }

            if (entry.stream.preceded_by(deps)) {
                return &entry;
            }

            if (!same_direction && entry.last_kind.load(std::memory_order_relaxed) == kind) {
                same_direction = &entry;
            }

            if (!any) {
                any = &entry;
            }
        }

        if (match_direction && same_direction) {
            return same_direction;
        }

        if (any) {
            return any;
        }
    }

    return &pool[cursor.fetch_add(1, std::memory_order_relaxed) % pool.size()];
}

}  // namespace kmm
