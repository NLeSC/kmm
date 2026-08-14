#include <algorithm>
#include <array>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "spdlog/spdlog.h"

#include "kmm/core/panic.hpp"
#include "kmm/runtime/device_data_streams.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

static constexpr size_t MAX_KINDS = 3;

struct StreamSlot {
    StreamKind kind;
    DeviceStreamId id;
    CUDAStream native_stream;
    uint64_t estimated_finish_time = 0;
    DeviceEventSet last_preds {};
    DeviceEvent last_event {};
    size_t active_users = 0;
    uint64_t last_acquired = 0;

    StreamSlot(StreamKind kind, DeviceStreamId id, CUDAStream native_stream) :
        kind(kind),
        id(id),
        native_stream(std::move(native_stream)) {}
};

struct DeviceDataStreams::Impl {
    std::array<std::array<std::vector<StreamSlot>, MAX_KINDS>, MAX_DEVICES> streams_per_device;
    ankerl::unordered_dense::map<DeviceStreamId, StreamSlot*> id_to_slot;
    std::map<DeviceEvent, uint64_t> estimated_finish_time;
    DeviceEventRegistry events;
    uint64_t acquire_counter = 0;

    Impl(DeviceEventRegistry events) : events(std::move(events)) {}
};

DeviceDataStreams::DeviceDataStreams(
    const SystemInfo& info,
    DeviceEventRegistry events,
    size_t num_d2d_streams,
    size_t num_h2d_streams,
    size_t num_d2h_streams
) :
    m_impl(std::make_unique<Impl>(events)) {
    for (size_t i = 0; i < info.num_devices(); i++) {
        auto* context = info.device(DeviceId(i)).context();
        std::array options = {
            std::make_tuple(StreamKind::DeviceToDevice, num_d2d_streams, "d2d"),
            std::make_tuple(StreamKind::HostToDevice, num_h2d_streams, "h2d"),
            std::make_tuple(StreamKind::DeviceToHost, num_d2h_streams, "d2h")};

        for (auto [kind, n, name] : options) {
            auto& slots = m_impl->streams_per_device[i][size_t(kind)];

            if (n <= 0) {
                throw std::runtime_error("number of streams must be positive");
            }

            for (size_t j = 0; j < n; j++) {
                auto label = fmt::format("gpu{}-{}-{}", i, name, j);

                auto stream = CUDAStream {context};
                auto id = events.register_stream(stream, label);
                slots.emplace_back(kind, id, std::move(stream));
            }

            for (auto& slot : slots) {
                m_impl->id_to_slot[slot.id] = &slot;
            }
        }
    }
}

DeviceDataStreams::DeviceDataStreams(DeviceDataStreams&&) noexcept = default;

DeviceDataStreams::~DeviceDataStreams() {
    for (auto& [id, slot] : m_impl->id_to_slot) {
        m_impl->events.unregister_stream(id);
    }
}

void DeviceDataStreams::make_progress() {
    auto it = m_impl->estimated_finish_time.begin();

    while (it != m_impl->estimated_finish_time.end()) {
        // not ready, exit now
        if (!m_impl->events.is_ready(it->first)) {
            break;
        }

        // erase the event
        it = m_impl->estimated_finish_time.erase(it);
    }

    for (auto& [id, slot] : m_impl->id_to_slot) {
        if (m_impl->events.is_ready(slot->last_event) && slot->active_users == 0) {
            slot->estimated_finish_time = 0;
            slot->last_acquired = 0;
        }
    }
}

static bool deps_equal_ignoring_stream(
    const DeviceEventRegistry& events,
    const DeviceEventSet& a,
    const DeviceEventSet& b,
    DeviceStreamId ignore_stream
) {
    for (const auto& event : a) {
        if (event.stream() == ignore_stream || events.is_ready(event)) {
            continue;
        }

        bool result = false;

        for (const auto& e : b) {
            if (event.precedes(e)) {
                result = true;
            }
        }

        if (!result) {
            return false;
        }
    }

    for (const auto& event : b) {
        if (event.stream() == ignore_stream || events.is_ready(event)) {
            continue;
        }

        bool result = false;

        for (const auto& e : a) {
            if (event.precedes(e)) {
                result = true;
            }
        }

        if (!result) {
            return false;
        }
    }

    return true;
}

DeviceStreamId DeviceDataStreams::acquire_stream(
    DeviceId device_id,
    StreamKind kind,
    const DeviceEventSet& deps
) {
    KMM_ASSERT(device_id.get() < MAX_DEVICES && size_t(kind) < MAX_KINDS);
    auto& slots = m_impl->streams_per_device[device_id.get()][size_t(kind)];

    uint64_t expected_start_time = 0;

    for (const auto& dep : deps) {
        // `lower_bound` lands exactly on `dep` if it has a recorded prediction; otherwise it
        // gives a starting point to scan forward for the next event on the same stream, whose
        // finish-time estimate is a valid (if possibly loose) upper bound on `dep`'s.
        auto it = m_impl->estimated_finish_time.lower_bound(dep);

        while (it != m_impl->estimated_finish_time.end() && it->first.stream() != dep.stream()) {
            ++it;
        }

        if (it == m_impl->estimated_finish_time.end() || it->first.stream() != dep.stream()) {
            continue;
        }

        // This event has already finished, we can remove it and ignore it.
        if (m_impl->events.is_ready(it->first)) {
            m_impl->estimated_finish_time.erase(it);
            continue;
        }

        expected_start_time = std::max(expected_start_time, it->second);
    }

    StreamSlot* best_slot = nullptr;
    std::tuple<size_t, bool, uint64_t, uint64_t, uint64_t> best_key;

    spdlog::trace(
        "selecting stream for {} (kind={}, deps={}, expected_start={})",
        device_id,
        kind,
        deps,
        expected_start_time
    );

    for (auto& slot : slots) {
        bool slot_has_affinity = deps.contains(slot.last_event);
        uint64_t slot_start_time = std::max(slot.estimated_finish_time, expected_start_time);
        uint64_t slot_slack = slot_start_time - slot.estimated_finish_time;

        // For H2D/D2H, the hardware has only one copy engine per direction, so separate
        // streams don't get parallelism. If this slot saw the exact same dependencies
        // last time, there's nothing to lose by reusing it
        if (kind != StreamKind::DeviceToDevice && !slot_has_affinity) {
            slot_has_affinity |=
                deps_equal_ignoring_stream(m_impl->events, slot.last_preds, deps, slot.id);
        }

        // Priority, in order:
        // - fewest active users,
        // - reuse the stream that produced the dependency,
        // - earliest start time,
        // - lowest idle time (previous finish time - next start time),
        // - longest since last selected.
        auto slot_key = std::make_tuple(
            slot.active_users,
            !slot_has_affinity,
            slot_start_time,
            slot_slack,
            slot.last_acquired
        );

        spdlog::trace(
            " - slot {}: last_event={}, users={}, affinity={}, start={}, slack={}",
            slot.id,
            slot.last_event,
            slot.active_users,
            slot_has_affinity,
            slot_start_time,
            slot_slack
        );

        if (best_slot == nullptr || slot_key < best_key) {
            best_slot = &slot;
            best_key = slot_key;
        }
    }

    KMM_ASSERT(best_slot);
    best_slot->active_users++;
    best_slot->last_preds = deps;
    best_slot->estimated_finish_time =
        std::max(best_slot->estimated_finish_time, expected_start_time);
    best_slot->last_acquired = ++m_impl->acquire_counter;

    // let the stream wait on the dependencies.
    m_impl->events.wait_on_event(best_slot->id, deps);

    spdlog::trace(
        "acquired data stream {} for {} (start_time={})",
        best_slot->id,
        kind,
        best_slot->estimated_finish_time
    );
    return best_slot->id;
}

DeviceEvent DeviceDataStreams::release_stream(DeviceStreamId stream_id, uint64_t cost) {
    auto it = m_impl->id_to_slot.find(stream_id);
    KMM_ASSERT(it != m_impl->id_to_slot.end());

    auto event = m_impl->events.record(stream_id);
    auto& slot = *it->second;
    slot.active_users--;
    slot.estimated_finish_time += cost;
    slot.last_event = event;
    m_impl->estimated_finish_time[event] = slot.estimated_finish_time;

    spdlog::trace(
        "released data stream {} for {} (finish_time={})",
        slot.id,
        slot.kind,
        slot.estimated_finish_time
    );

    return event;
}

std::ostream& operator<<(std::ostream& stream, StreamKind kind) {
    switch (kind) {
        case StreamKind::DeviceToDevice:
            return stream << "DeviceToDevice";
            break;
        case StreamKind::HostToDevice:
            return stream << "HostToDevice";
            break;
        case StreamKind::DeviceToHost:
            return stream << "DeviceToHost";
            break;
        default:
            return stream << "Unknown";
    }
}

}  // namespace kmm
