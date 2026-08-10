#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <deque>
#include <mutex>
#include <queue>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

// A pool of reusable, disabled-timing `CUevent`s for a single `CUcontext`. Streams sharing a
// context share a pool, since the events themselves are not tied to a specific stream: they are
// only ever recorded onto whichever stream needs a fresh event at that moment.
struct EventPool: reference_count<EventPool> {
    KMM_NOT_COPYABLE_OR_MOVABLE(EventPool)

  public:
    explicit EventPool(CUcontext context) : m_context(context) {}
    ~EventPool();

    CUevent pop();
    void push(CUevent event);

  private:
    CUcontext m_context;
    std::mutex m_mutex;
    std::vector<CUevent> m_events;
};

EventPool::~EventPool() {
    CUDAContextGuard guard {m_context};

    for (CUevent event : m_events) {
        KMM_CUDA_CHECK(cuEventSynchronize(event));
        KMM_CUDA_CHECK(cuEventDestroy(event));
    }
}

CUevent EventPool::pop() {
    std::lock_guard<std::mutex> guard(m_mutex);

    if (!m_events.empty()) {
        CUevent event = m_events.back();
        m_events.pop_back();
        return event;
    }

    CUDAContextGuard context_guard {m_context};
    CUevent event;
    KMM_CUDA_CHECK(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    return event;
}

void EventPool::push(CUevent event) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_events.push_back(event);
}

// Source of each `DeviceStreamImpl::id`, the cheap per-stream key `PredecessorsSet` is keyed by.
// Starts at 1 so that 0 stays free as a sentinel for "no stream" if ever needed.
std::atomic<uint64_t> g_global_stream_count {1};

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

// Holds the state behind a `DeviceStream`. Kept deliberately thin: only the handful of helpers
// that are genuinely shared across multiple `DeviceStream::` methods live here (`poll`,
// `is_ready`, `try_resolve_event_locked`, `self_locked`); everything else is just data that
// `DeviceStream::`'s methods manipulate directly.
struct DeviceStreamImpl: reference_count<DeviceStreamImpl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceStreamImpl)

  public:
    DeviceStreamImpl(
        CUcontext ctx,
        CUstream strm,
        refcnt_ptr<EventPool> pool,
        bool destroy_if_done
    ) :
        context_id(ctx),
        context(ctx),
        stream_id(strm),
        stream(strm),
        destroy_if_done(destroy_if_done),
        m_pool(std::move(pool)) {}

    ~DeviceStreamImpl() {
        KMM_CUDA_CHECK(cuStreamSynchronize(stream));

        for (CUevent event : m_pending_events) {
            KMM_CUDA_CHECK(cuEventSynchronize(event));
            KMM_CUDA_CHECK(cuEventDestroy(event));
        }

        while (!m_callback_heap.empty()) {
            auto callback = std::move(m_callback_heap.top().callback);
            m_callback_heap.pop();
            callback.notify_and_clear();
        }

        if (destroy_if_done) {
            KMM_CUDA_CHECK(cuStreamDestroy(stream));
        }
    }

    const uint64_t id = g_global_stream_count++;
    const cuda_context_id context_id;
    const CUcontext context;
    const cuda_stream_id stream_id;
    const CUstream stream;
    const bool destroy_if_done;

    std::atomic<uint64_t> m_first_pending_event {1};

    // Index of the most recently recorded event on this stream, or 0 if none has been
    // recorded yet. Updated in `record_event` under `m_mutex`, but read lock-free.
    std::atomic<uint64_t> m_last_pending_event {0};

    mutable std::mutex m_mutex;
    refcnt_ptr<EventPool> m_pool;
    std::deque<CUevent> m_pending_events;
    std::priority_queue<EventCallback> m_callback_heap;

    // Maps stream id => event id for the events that this stream has waited for. This allows
    // one to check if any event on another stream precedes the events on this stream.
    ankerl::unordered_dense::map<uint64_t, uint64_t> m_preceded_by;

    // True if `m_preceded_by` is still exactly as it was when the most recently recorded event
    // was recorded (i.e., no `wait_on_event` has touched it since). Only in that case is it valid
    // for another stream to assume that  `m_preceded_by` is for the latest event.
    bool m_preceded_by_is_current = true;

    bool is_ready(uint64_t event_id) const noexcept {
        return event_id < m_first_pending_event.load(std::memory_order_acquire);
    }

    bool is_latest(uint64_t event_id) const noexcept {
        return event_id == m_last_pending_event.load(std::memory_order_acquire);
    }

    // Precondition: caller holds `m_mutex`.
    CUevent try_resolve_event_locked(uint64_t event_id) {
        auto first_pending = m_first_pending_event.load(std::memory_order_acquire);

        // event is ready
        if (event_id < first_pending) {
            return nullptr;
        }

        uint64_t offset = event_id - first_pending;
        KMM_ASSERT(offset < m_pending_events.size());
        return m_pending_events[offset];
    }
};

KMM_REFCNT_TRAITS_IMPL(DeviceStreamImpl)

DeviceStream::DeviceStream(refcnt_ptr<Impl> impl) noexcept : m_impl(std::move(impl)) {}

DeviceStream DeviceStream::create(CUcontext context, CUstream stream, bool destroy_if_done) {
    cuda_context_id id(context);
    refcnt_ptr<EventPool> pool = make_refcnt<EventPool>(context);
    return DeviceStream {make_refcnt<Impl>(context, stream, std::move(pool), destroy_if_done)};
}

DeviceEvent DeviceStream::record_event() const {
    KMM_ASSERT(m_impl);
    std::lock_guard<std::mutex> guard(m_impl->m_mutex);

    CUevent event = m_impl->m_pool->pop();

    try {
        KMM_CUDA_CHECK(cuEventRecord(event, m_impl->stream));
    } catch (...) {
        // don't leak the event!
        m_impl->m_pool->push(event);
        throw;
    }

    uint64_t index = m_impl->m_first_pending_event.load(std::memory_order_relaxed)
        + m_impl->m_pending_events.size();
    spdlog::debug("record on stream {} new event {}", m_impl->stream_id.get(), index);

    m_impl->m_pending_events.push_back(event);
    m_impl->m_last_pending_event.store(index, std::memory_order_release);
    m_impl->m_preceded_by_is_current = true;

    return DeviceEvent {*this, index};
}

bool DeviceStream::is_ready() const noexcept {
    KMM_ASSERT(m_impl);
    CUresult result = cuStreamQuery(m_impl->stream);

    if (result == CUDA_ERROR_NOT_READY) {
        return false;
    }

    KMM_CUDA_CHECK(result);
    return true;
}

void DeviceStream::make_progress() const {
    KMM_ASSERT(m_impl);
    std::lock_guard<std::mutex> guard(m_impl->m_mutex);

    while (!m_impl->m_pending_events.empty()) {
        CUevent event = m_impl->m_pending_events.front();
        CUresult result = cuEventQuery(event);

        if (result == CUDA_ERROR_NOT_READY) {
            break;
        }

        KMM_CUDA_CHECK(result);

        m_impl->m_pending_events.pop_front();
        m_impl->m_pool->push(event);
        m_impl->m_first_pending_event.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t ready_before = m_impl->m_first_pending_event.load(std::memory_order_relaxed);

    while (!m_impl->m_callback_heap.empty()
           && m_impl->m_callback_heap.top().event_index < ready_before) {
        m_impl->m_callback_heap.top().callback.notify();
        m_impl->m_callback_heap.pop();
    }
}

void DeviceStream::attach_callback(NotifyHandle callback) const {
    KMM_ASSERT(m_impl);

    {
        std::lock_guard<std::mutex> guard(m_impl->m_mutex);

        if (!m_impl->m_pending_events.empty()) {
            uint64_t tail_index = m_impl->m_last_pending_event.load(std::memory_order_relaxed);

            m_impl->m_callback_heap.emplace(tail_index, std::move(callback));
            return;
        }
    }

    callback.notify_and_clear();
}

cuda_stream_id DeviceStream::id() const {
    KMM_ASSERT(m_impl);
    return m_impl->stream_id;
}

CUstream DeviceStream::get() const {
    return m_impl ? m_impl->stream : nullptr;
}

CUcontext DeviceStream::context() const {
    return m_impl ? m_impl->context : nullptr;
}

void DeviceStream::wait_on_default_stream() const {
    KMM_ASSERT(m_impl);
    std::lock_guard<std::mutex> guard(m_impl->m_mutex);

    CUevent event = m_impl->m_pool->pop();

    // The next method may throw. We put the event back into the pool now. Since we keep the
    // lock, the event cannot be reclaimed while we perform the CUDA operations.
    m_impl->m_pool->push(event);

    KMM_CUDA_CHECK(cuEventRecord(event, nullptr));
    KMM_CUDA_CHECK(cuStreamWaitEvent(m_impl->stream, event, 0));
}

void DeviceStream::wait_on_event(CUevent event) const {
    KMM_ASSERT(m_impl);
    KMM_CUDA_CHECK(cuStreamWaitEvent(m_impl->stream, event, 0));
}

void DeviceStream::wait_on_event(const DeviceEvent& event) const {
    KMM_ASSERT(m_impl);

    if (event.is_null()) {
        return;
    }

    // Work recorded on a stream is always ordered after events already recorded on that same
    // stream, so there is nothing to do here. This also avoids locking this stream's mutex
    // against itself below (source and destination would be the same mutex).
    if (event.stream() == *this) {
        return;
    }

    DeviceStreamImpl& src_state = *event.stream().m_impl;
    uint64_t src_event = event.index();

    std::scoped_lock guard {m_impl->m_mutex, src_state.m_mutex};
    CUevent handle = src_state.try_resolve_event_locked(src_event);

    if (handle == nullptr) {
        return;
    }

    KMM_CUDA_CHECK(cuStreamWaitEvent(m_impl->stream, handle, 0));
    spdlog::debug(
        "stream {} must wait on event {}:{}",
        m_impl->stream_id.get(),
        src_state.stream_id.get(),
        src_event
    );

    // This stream now directly waits on `src_event`, so it is always a valid predecessor of
    // any future work recorded on this stream.
    m_impl->m_preceded_by.emplace(src_state.id, src_event);

    // If `src_event` is the latest event recorded on `src` and nothing has touched
    // `src.m_preceded_by` since, then everything in it was already true by the time `src_event`
    // was recorded. This stream now transitively waits on all of it too, so it can be inherited
    // wholesale instead of only being discovered one hop at a time on future waits.
    if (src_event == src_state.m_last_pending_event.load(std::memory_order_relaxed)
        && src_state.m_preceded_by_is_current) {
        for (const auto& [stream_id, event_id] : src_state.m_preceded_by) {
            auto& value = m_impl->m_preceded_by[stream_id];
            value = std::max(value, event_id);
        }
    }

    m_impl->m_preceded_by_is_current = false;
}

void DeviceStream::wait_on_events(const DeviceEventSet& events) const {
    KMM_ASSERT(m_impl);

    for (const auto& event : events) {
        wait_on_event(event);
    }
}

void DeviceStream::synchronize() const {
    KMM_ASSERT(m_impl);
    KMM_CUDA_CHECK(cuStreamSynchronize(m_impl->stream));
    make_progress();
}

bool DeviceStream::preceded_by(const DeviceEvent& src) const {
    KMM_ASSERT(m_impl);

    if (src.is_null()) {
        return true;
    }

    if (src.stream() == *this) {
        return true;
    }

    if (src.is_ready()) {
        return true;
    }

    std::lock_guard<std::mutex> guard(m_impl->m_mutex);
    return m_impl->m_preceded_by[src.stream().m_impl->id] <= src.index();
}

bool DeviceStream::preceded_by(const DeviceEventSet& src) const {
    KMM_ASSERT(m_impl);

    for (const auto& event : src) {
        if (!preceded_by(event)) {
            return false;
        }
    }

    return true;
}

bool DeviceStream::is_latest_in(const DeviceEventSet& deps) const {
    KMM_ASSERT(m_impl);

    for (const auto& event : deps) {
        if (event.stream() == *this && this->is_latest(event.index())) {
            return true;
        }
    }

    return false;
}

void DeviceStream::attach_callback(uint64_t event_id, NotifyHandle callback) const {
    KMM_ASSERT(m_impl);

    {
        std::lock_guard<std::mutex> guard(m_impl->m_mutex);

        if (event_id >= m_impl->m_first_pending_event.load(std::memory_order_relaxed)) {
            m_impl->m_callback_heap.emplace(event_id, std::move(callback));
            return;
        }
    }

    callback.notify_and_clear();
}

void DeviceStream::synchronize(uint64_t event_id) const {
    KMM_ASSERT(m_impl);

    CUevent handle;

    {
        std::lock_guard<std::mutex> guard(m_impl->m_mutex);

        if (m_impl->is_ready(event_id)) {
            return;
        }

        handle = m_impl->try_resolve_event_locked(event_id);
    }

    if (handle != nullptr) {
        KMM_CUDA_CHECK(cuEventSynchronize(handle));
    }

    make_progress();
}

bool DeviceStream::is_ready(uint64_t event_id) const {
    KMM_ASSERT(m_impl);
    return m_impl->is_ready(event_id);
}

bool DeviceStream::is_latest(uint64_t event_id) const {
    KMM_ASSERT(m_impl);
    return m_impl->is_latest(event_id);
}

DeviceEvent DeviceStream::with_stream(const DeviceEventSet& pred, function_ref<void(CUstream)> fun)
    const {
    KMM_ASSERT(m_impl);
    wait_on_events(pred);
    fun(m_impl->stream);
    return record_event();
}

bool DeviceStream::with_event(uint64_t event_id, function_ref<void(CUstream, CUevent)> callback)
    const {
    KMM_ASSERT(m_impl);

    std::lock_guard<std::mutex> guard(m_impl->m_mutex);

    if (CUevent handle = m_impl->try_resolve_event_locked(event_id)) {
        callback(m_impl->stream, handle);
        return true;
    } else {
        return false;
    }
}

}  // namespace kmm
