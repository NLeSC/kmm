#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <mutex>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "fmt/chrono.h"
#include "fmt/ostream.h"
#include "spdlog/spdlog.h"

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/device_event_registry.hpp"
#include "kmm/runtime/device_stream.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

class PrecedenceVector {
  public:
    void update(DeviceEvent e) {
        auto& it = m_entries[e.stream()];
        it = std::max(e.index(), it);
    }

    bool contains(DeviceEvent e) const {
        auto it = m_entries.find(e.stream());
        return it != m_entries.end() && it->second >= e.index();
    }

    void merge(const PrecedenceVector& that) {
        for (auto entry : that.m_entries) {
            auto& it = m_entries[entry.first];
            it = std::max(entry.second, it);
        }
    }

    DeviceEventSet to_set() {
        DeviceEventSet result;
        result.m_events.resize(m_entries.size());
        size_t i = 0;

        for (auto entry : m_entries) {
            result.m_events[i++] = DeviceEvent {entry.first, entry.second};
        }

        return result;
    }

  private:
    ankerl::unordered_dense::map<DeviceStreamId, uint64_t> m_entries;
};

struct EventCallback {
    EventCallback(uint64_t event_index, NotifyHandle callback) :
        event_index(event_index),
        callback(std::move(callback)) {}

    friend bool operator<(const EventCallback& a, const EventCallback& b) {
        return a.event_index > b.event_index;
    }

    uint64_t event_index;
    mutable NotifyHandle callback;
};

struct StreamState {
    KMM_NOT_COPYABLE_OR_MOVABLE(StreamState)

  public:
    StreamState(DeviceStreamId id, GPUStreamRef stream_ref, std::string name) :
        id(id),
        context(stream_ref.context()),
        stream(stream_ref.stream()),
        stream_key(stream_ref.stream_id()),
        name(std::move(name)) {}

    friend std::ostream& operator<<(std::ostream& stream, const StreamState& state) {
        stream << state.id;

        if (!state.name.empty()) {
            stream << " \"" << state.name << "\"";
        }

        return stream;
    }

    ~StreamState() {
        GPUContextGuard guard {context};

        for (const auto& entry : pending_events) {
            KMM_GPU_CHECK(gpuEventSynchronize(entry.second));
            KMM_GPU_CHECK(gpuEventDestroy(entry.second));
        }

        for (GPUEvent event : free_events) {
            KMM_GPU_CHECK(gpuEventDestroy(event));
        }

        while (!callbacks.empty()) {
            auto callback = std::move(callbacks.top().callback);
            callbacks.pop();
            callback.notify_and_clear();
        }
    }

    // Precondition: caller holds `mutex`.
    GPUEvent pop_event_locked() {
        if (!free_events.empty()) {
            GPUEvent event = free_events.back();
            free_events.pop_back();
            return event;
        }

        GPUContextGuard guard {context};
        GPUEvent event;
        KMM_GPU_CHECK(gpuEventCreate(&event));
        return event;
    }

    // Precondition: caller holds `mutex`.
    GPUEvent try_resolve_event_locked(uint64_t event_id) const {
        uint64_t first_pending = first_pending_event.load(std::memory_order_acquire);

        if (event_id < first_pending) {
            return nullptr;
        }

        // Event ids are globally unique (shared across all streams), so this stream's own
        // `pending_events` no longer line up with a contiguous range: find the id by binary
        // search instead of by direct offset.
        auto it = std::lower_bound(
            pending_events.begin(),
            pending_events.end(),
            event_id,
            [](const auto& entry, uint64_t id) { return entry.first < id; }
        );

        KMM_ASSERT(it != pending_events.end() && it->first == event_id);
        return it->second;
    }

    // Precondition: caller holds `mutex`. Drawing the id here (rather than before the lock is
    // acquired) guarantees that id order always matches the order events are pushed onto this
    // stream's `pending_events`, even if `record()` is ever called concurrently for this stream.
    uint64_t record_locked(std::atomic<uint64_t>& next_event_id) {
        if (released) {
            throw std::runtime_error("cannot record an event on a stream that has been unregistered"
            );
        }

        GPUEvent event = pop_event_locked();

        try {
            KMM_GPU_CHECK(gpuEventRecord(event, stream));
        } catch (...) {
            // don't leak the event!
            free_events.push_back(event);
            throw;
        }

        uint64_t new_event = next_event_id.fetch_add(1, std::memory_order_relaxed);
        spdlog::debug("recorded new event {} on stream {}", DeviceEvent(id, new_event), id);

        // if there are not pending events, we can quickly check if this new event also completed
        // immediately. This is possible if the stream is idle, and we just recorded after nothing.
        bool is_complete = pending_events.empty() && gpuEventQuery(event) == GPU_SUCCESS && false;

        if (is_complete) {
            free_events.push_back(event);
            first_pending_event.store(new_event + 1, std::memory_order_release);
            spdlog::debug("completed event {} on stream {}", DeviceEvent(id, new_event), id);
        } else {
            pending_events.push_back({new_event, event});
        }

        last_pending_event.store(new_event, std::memory_order_release);

        // `preceded_by` (plus this fresh self-entry) now exactly describes the dependencies of
        // the event just recorded, so it is back in sync with `last_pending_event`.
        preceded_by.update(DeviceEvent {id, new_event});
        preceded_by_is_last = true;

        return new_event;
    }

    // Reclaims events that have completed back into the free list and fires callbacks that were
    // waiting on them. Precondition: caller holds `state.mutex`.
    void make_progress() {
        while (!pending_events.empty()) {
            auto [completed_id, event] = pending_events.front();
            GPUResult result = gpuEventQuery(event);

            if (result == GPU_ERROR_NOT_READY) {
                break;
            }

            KMM_GPU_CHECK(result);

            pending_events.pop_front();
            free_events.push_back(event);

            // A stream's own events always complete in the order they were recorded, so once
            // `completed_id` is done, everything this stream recorded before it is done too.
            first_pending_event.store(completed_id + 1, std::memory_order_relaxed);
            spdlog::debug("completed event {} on stream {}", DeviceEvent(id, completed_id), id);
        }

        uint64_t ready_before = first_pending_event.load(std::memory_order_relaxed);

        while (!callbacks.empty() && callbacks.top().event_index < ready_before) {
            callbacks.top().callback.notify();
            callbacks.pop();
        }
    }

    void trim_event_pool_locked() {
        GPUContextGuard context_guard {context};

        for (GPUEvent event : free_events) {
            KMM_GPU_CHECK(gpuEventSynchronize(event));
            KMM_GPU_CHECK(gpuEventDestroy(event));
        }

        free_events.clear();
    }

    const DeviceStreamId id;
    const GPUContext context;
    const GPUStream stream;

    // Canonical driver-level identity of `stream`, cached at registration time so that
    // `lookup_stream` can compare against it without a fresh `gpuStreamGetId` call per candidate.
    const GPUStreamId stream_key;

    // Optional human-readable name, included in log messages that refer to this stream.
    const std::string name;

    mutable std::mutex mutex;
    std::vector<GPUEvent> free_events;
    // Pairs of (globally unique event id, GPUEvent), ordered by id (equivalently, by the order
    // in which they were recorded on this stream).
    std::deque<std::pair<uint64_t, GPUEvent>> pending_events;
    std::priority_queue<EventCallback> callbacks;

    std::atomic<uint64_t> first_pending_event {1};
    std::atomic<uint64_t> last_pending_event {0};

    // Tracks what this stream has waited for. This allows one to check if some event on another
    // stream already precedes this stream's future work. `preceded_by_is_last` indicates if
    // the precedence vector corresponds to the last recorded event on this stream or not.
    PrecedenceVector preceded_by;
    bool preceded_by_is_last = false;

    // Indicates if `unregister_stream` has been called. No new events can be recorded.
    bool released = false;
};

}  // namespace kmm

template<>
struct fmt::formatter<kmm::StreamState>: fmt::ostream_formatter {};

namespace kmm {

struct DeviceEventRegistry::Impl: reference_count<Impl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(Impl)

  public:
    Impl() = default;

    ~Impl() {
        for (auto& slot : streams) {
            delete slot.load(std::memory_order_relaxed);
        }
    }

    StreamState* stream_opt(DeviceStreamId id) {
        return id.is_null() ? nullptr : streams[id.get()].load(std::memory_order_acquire);
    }

    StreamState& stream(DeviceStreamId id) {
        StreamState* state = stream_opt(id);
        KMM_ASSERT(state != nullptr);
        return *state;
    }

    std::array<std::atomic<StreamState*>, MAX_DEVICE_STREAMS> streams {};

    // Shared across all streams so that every recorded event gets a globally unique id, which
    // makes events unambiguous when printed/logged (instead of resetting per stream).
    std::atomic<uint64_t> next_event_id {1};
};

KMM_REFCNT_TRAITS_IMPL(DeviceEventRegistry::Impl)

DeviceEventRegistry::DeviceEventRegistry() : m_impl(make_refcnt<Impl>()) {}

DeviceStreamId DeviceEventRegistry::register_stream(GPUStreamRef stream_ref, std::string name)
    const {
    for (uint64_t i = 0; i < MAX_DEVICE_STREAMS; i++) {
        auto id = DeviceStreamId(i);
        auto* state = new StreamState(id, stream_ref, name);
        StreamState* expected = nullptr;

        if (m_impl->streams[i].compare_exchange_strong(
                expected,
                state,
                std::memory_order_release,
                std::memory_order_relaxed
            )) {
            spdlog::debug("registered new stream {} from {}", *state, state->stream_key);
            return id;
        }

        delete state;
    }

    throw std::runtime_error(
        "cannot register stream: the maximum number of streams (MAX_DEVICE_STREAMS) has been reached"
    );
}

void DeviceEventRegistry::unregister_stream(DeviceStreamId stream_id) const {
    StreamState* state = m_impl->streams[stream_id.get()].load(std::memory_order_acquire);

    // no stream registered, just ignore
    if (state == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> guard(state->mutex);

    // stream was already released, skip it
    if (state->released) {
        return;
    }

    // Block until every event ever recorded on this stream has completed, so that once this
    // returns, nothing is still relying on the stream being alive. Held under `mutex` for the
    // whole check-and-act so a concurrent call for the same stream either blocks here and then
    // observes `released` and returns, or never gets in until this one has fully finished.
    KMM_GPU_CHECK(gpuStreamSynchronize(state->stream));

    state->make_progress();
    state->trim_event_pool_locked();
    state->released = true;

    spdlog::debug("unregistered new stream {} from GPU stream {}", state->id, state->stream_key);
}

std::optional<DeviceStreamId> DeviceEventRegistry::lookup_stream(GPUStreamId target) const {
    for (uint64_t i = 0; i < MAX_DEVICE_STREAMS; i++) {
        StreamState* state = m_impl->streams[i].load(std::memory_order_acquire);

        if (state != nullptr && state->stream_key == target) {
            return DeviceStreamId(i);
        }
    }

    return std::nullopt;
}

DeviceStreamId DeviceEventRegistry::lookup_or_register_stream(GPUStreamRef stream_ref) const {
    GPUStreamId target = stream_ref.stream_id();

    for (uint64_t i = 0; i < MAX_DEVICE_STREAMS; i++) {
        StreamState* state = m_impl->streams[i].load(std::memory_order_acquire);

        if (state == nullptr) {
            auto id = DeviceStreamId(i);
            auto* new_state = new StreamState(id, stream_ref, "");

            if (m_impl->streams[i].compare_exchange_strong(
                    state,
                    new_state,
                    std::memory_order_release,
                    std::memory_order_relaxed
                )) {
                spdlog::debug(
                    "registered new stream {} from GPU stream {}",
                    new_state->id,
                    new_state->stream_key
                );
                return id;
            }

            delete new_state;
        } else if (state->stream_key == target) {
            return DeviceStreamId(i);
        }
    }

    throw std::runtime_error(
        "cannot register stream: the maximum number of streams (MAX_DEVICE_STREAMS) has been reached"
    );
}

void DeviceEventRegistry::shutdown() const {
    for (uint64_t i = 0; i < MAX_DEVICE_STREAMS; i++) {
        unregister_stream(DeviceStreamId(i));
    }
}

GPUStream DeviceEventRegistry::get(DeviceStreamId stream_id) const {
    return m_impl->stream(stream_id).stream;
}

GPUContext DeviceEventRegistry::context(DeviceStreamId stream_id) const {
    return m_impl->stream(stream_id).context;
}

DeviceStream DeviceEventRegistry::stream(DeviceStreamId stream_id) const {
    return DeviceStream(*this, stream_id);
}

bool DeviceEventRegistry::has_context(DeviceStreamId stream_id, GPUContextId context_id) const {
    if (auto* state = m_impl->stream_opt(stream_id)) {
        return state->stream_key.context() == context_id;
    } else {
        return false;
    }
}

DeviceEvent DeviceEventRegistry::record(DeviceStreamId stream_id) const {
    StreamState& state = m_impl->stream(stream_id);
    std::lock_guard<std::mutex> guard(state.mutex);
    auto index = state.record_locked(m_impl->next_event_id);
    return DeviceEvent {stream_id, index};
}

void DeviceEventRegistry::wait_on_event(DeviceStreamId stream_id, DeviceEvent event) const {
    if (event.is_null()) {
        return;
    }

    // Work recorded on a stream is always ordered after events already recorded on that same
    // stream, so there is nothing to do here. This also avoids locking this stream's mutex
    // against itself below (source and destination would be the same mutex).
    if (event.stream() == stream_id) {
        return;
    }

    StreamState& dst = m_impl->stream(stream_id);
    StreamState& src = m_impl->stream(event.stream());
    uint64_t src_event = event.index();

    std::scoped_lock guard {dst.mutex, src.mutex};

    // `dst` already waited (directly or transitively) on `event`
    if (dst.preceded_by.contains(event)) {
        return;
    }

    GPUEvent handle = src.try_resolve_event_locked(src_event);

    if (handle == nullptr) {
        return;
    }

    KMM_GPU_CHECK(gpuStreamWaitEvent(dst.stream, handle));
    spdlog::debug("stream {} must wait on event {}", dst.id, event);

    // This stream now directly waits on `src_event`, so it is always a valid predecessor of
    // any future work recorded on this stream. This does not record a new event on `dst`, so
    // `preceded_by` now describes dependencies for whatever gets recorded next rather than for
    // `last_pending_event`, which makes it out of sync until the next `record_locked()` call.
    dst.preceded_by.update(event);
    dst.preceded_by_is_last = false;

    // We can only fold in everything that precedes `src_event` if `src_event` is `src`'s latest
    // event and `src` itself has complete knowledge of what precedes it.

    /* TODO: not sure about this? */
    //    uint64_t src_last = src.last_pending_event.load(std::memory_order_relaxed);
    //    if (src_last == src_event && src.preceded_by_is_last) {
    //        dst.preceded_by.merge(src.preceded_by);
    //    }
}

void DeviceEventRegistry::wait_on_event(DeviceStreamId stream_id, const DeviceEventSet& events)
    const {
    for (const auto& event : events) {
        wait_on_event(stream_id, event);
    }
}

void DeviceEventRegistry::wait_on_event(GPUStream stream, DeviceEvent event) const {
    if (event.is_null()) {
        return;
    }

    StreamState& src = m_impl->stream(event.stream());
    uint64_t src_event = event.index();

    std::scoped_lock guard {src.mutex};
    GPUEvent handle = src.try_resolve_event_locked(src_event);

    if (handle == nullptr) {
        return;
    }

    KMM_GPU_CHECK(gpuStreamWaitEvent(stream, handle));
}

void DeviceEventRegistry::wait_on_event(GPUStream stream, const DeviceEventSet& events) const {
    for (const auto& event : events) {
        wait_on_event(stream, event);
    }
}

void DeviceEventRegistry::wait_on_default_stream(DeviceStreamId stream_id) const {
    StreamState& dst = m_impl->stream(stream_id);
    GPUContextGuard guard {dst.context};

    GPUEvent event;
    KMM_GPU_CHECK(gpuEventCreate(&event));

    try {
        // `nullptr` refers to the CUDA legacy default stream in the current context.
        KMM_GPU_CHECK(gpuEventRecord(event, nullptr));
        KMM_GPU_CHECK(gpuStreamWaitEvent(dst.stream, event));
    } catch (...) {
        KMM_GPU_CHECK(gpuEventDestroy(event));
        throw;
    }

    KMM_GPU_CHECK(gpuEventDestroy(event));
}

bool DeviceEventRegistry::is_ready(DeviceStreamId stream_id) const {
    if (stream_id.is_null()) {
        return true;
    }

    StreamState& state = m_impl->stream(stream_id);
    GPUResult result = gpuStreamQuery(state.stream);

    if (result == GPU_ERROR_NOT_READY) {
        return false;
    }

    KMM_GPU_CHECK(result);
    return true;
}

bool DeviceEventRegistry::is_ready(DeviceEvent event) const {
    if (event.is_null()) {
        return true;
    }

    StreamState& state = m_impl->stream(event.stream());
    return event.index() < state.first_pending_event.load(std::memory_order_acquire);
}

bool DeviceEventRegistry::is_ready(const DeviceEventSet& events) const {
    for (const auto& e : events) {
        if (!is_ready(e)) {
            return false;
        }
    }

    return true;
}

bool DeviceEventRegistry::is_latest(DeviceEvent event) const {
    if (event.is_null()) {
        return false;
    }

    StreamState& state = m_impl->stream(event.stream());
    return event.index() == state.last_pending_event.load(std::memory_order_acquire);
}

bool DeviceEventRegistry::is_latest_in(DeviceStreamId stream_id, const DeviceEventSet& deps) const {
    if (stream_id.is_null()) {
        return false;
    }

    for (const auto& event : deps) {
        if (event.stream() == stream_id && is_latest(event)) {
            return true;
        }
    }

    return false;
}

DeviceEvent DeviceEventRegistry::latest_event(DeviceStreamId stream_id) const {
    if (stream_id.is_null()) {
        return DeviceEvent::null();
    }

    StreamState& state = m_impl->stream(stream_id);
    uint64_t index = state.last_pending_event.load(std::memory_order_acquire);

    if (index == 0) {
        return DeviceEvent {};
    }

    return DeviceEvent {stream_id, index};
}

DeviceEventSet DeviceEventRegistry::snapshot(DeviceStreamId stream_id) const {
    if (stream_id.is_null()) {
        return {};
    }

    StreamState& state = m_impl->stream(stream_id);
    DeviceEventSet result;

    std::lock_guard<std::mutex> guard(state.mutex);
    return state.preceded_by.to_set();
}

void DeviceEventRegistry::synchronize(DeviceStreamId stream_id) const {
    if (stream_id.is_null()) {
        return;
    }

    StreamState& state = m_impl->stream(stream_id);

    auto before = std::chrono::system_clock::now();
    KMM_GPU_CHECK(gpuStreamSynchronize(state.stream));
    auto after = std::chrono::system_clock::now();

    auto duration = after - before;
    if (duration > std::chrono::milliseconds(1)) {
        spdlog::warn("waited for {} to synchronize with stream {}", duration, stream_id);
    }

    std::lock_guard<std::mutex> guard(state.mutex);
    state.make_progress();
}

void DeviceEventRegistry::synchronize(DeviceEvent event) const {
    if (event.is_null()) {
        return;
    }

    StreamState& state = m_impl->stream(event.stream());
    GPUEvent handle = nullptr;

    {
        std::lock_guard<std::mutex> guard(state.mutex);

        if (event.index() >= state.first_pending_event.load(std::memory_order_relaxed)) {
            handle = state.try_resolve_event_locked(event.index());
        }
    }

    if (handle == nullptr) {
        return;
    }

    auto before = std::chrono::system_clock::now();
    KMM_GPU_CHECK(gpuEventSynchronize(handle));
    auto after = std::chrono::system_clock::now();

    auto duration = after - before;
    if (duration > std::chrono::milliseconds(1)) {
        spdlog::warn("waited for {} to synchronize with event {}", duration, event);
    }

    std::lock_guard<std::mutex> guard(state.mutex);
    state.make_progress();
}

void DeviceEventRegistry::synchronize(const DeviceEventSet& events) const {
    for (const auto& e : events) {
        synchronize(e);
    }
}

void DeviceEventRegistry::synchronize_all() const {
    for (uint64_t i = 0; i < MAX_DEVICE_STREAMS; i++) {
        StreamState* state = m_impl->streams[i].load(std::memory_order_acquire);

        if (state != nullptr) {
            synchronize(DeviceStreamId(i));
        }
    }
}

void DeviceEventRegistry::attach_callback(DeviceEvent event, NotifyHandle callback) const {
    if (event.is_null()) {
        callback.notify_and_clear();
        return;
    }

    StreamState& state = m_impl->stream(event.stream());

    {
        std::lock_guard<std::mutex> guard(state.mutex);

        if (event.index() >= state.first_pending_event.load(std::memory_order_relaxed)) {
            state.callbacks.emplace(event.index(), std::move(callback));
            return;
        }
    }

    callback.notify_and_clear();
}

void DeviceEventRegistry::make_progress() const {
    for (auto& slot : m_impl->streams) {
        StreamState* state = slot.load(std::memory_order_acquire);

        if (state == nullptr) {
            continue;
        }

        std::lock_guard<std::mutex> guard(state->mutex);
        state->make_progress();
    }
}

bool DeviceEventRegistry::is_all_ready() const {
    for (auto& slot : m_impl->streams) {
        StreamState* state = slot.load(std::memory_order_acquire);

        if (state == nullptr) {
            continue;
        }

        std::lock_guard<std::mutex> guard(state->mutex);

        if (!state->pending_events.empty()) {
            return false;
        }
    }

    return true;
}

void DeviceEventRegistry::trim_event_pool(DeviceStreamId stream_id) const {
    if (stream_id.is_null()) {
        return;
    }

    StreamState& state = m_impl->stream(stream_id);
    std::lock_guard<std::mutex> guard(state.mutex);
    state.trim_event_pool_locked();
}

void DeviceEventRegistry::trim_event_pool() const {
    for (uint64_t i = 0; i < MAX_DEVICE_STREAMS; i++) {
        StreamState* state = m_impl->streams[i].load(std::memory_order_acquire);

        if (state != nullptr) {
            trim_event_pool(DeviceStreamId(i));
        }
    }
}

bool DeviceEventRegistry::precedes(const DeviceEvent& a, const DeviceStreamId& b) const {
    if (a.is_null() || b.is_null() || a.stream() == b) {
        return true;
    }

    if (is_ready(a)) {
        return true;
    }

    StreamState& dst = m_impl->stream(b);
    std::lock_guard<std::mutex> guard(dst.mutex);
    return dst.preceded_by.contains(a);
}

bool DeviceEventRegistry::precedes(const DeviceEventSet& a, const DeviceStreamId& b) const {
    for (const auto& event : a) {
        if (!precedes(event, b)) {
            return false;
        }
    }

    return true;
}

}  // namespace kmm
