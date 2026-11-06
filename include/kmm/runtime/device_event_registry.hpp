#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class DeviceStream;

/**
 * The event registry is used to keep track of all GPU events that have been recorded.
 */
class DeviceEventRegistry {
  public:
    DeviceEventRegistry();

    /**
     * Register a new stream with the event registry. The returned identifier can be used
     * to register events or block the stream on existing events. The optional `name` is
     * included in log messages that refer to this stream.
     */
    DeviceStreamId register_stream(GPUStreamRef stream_ref, std::string name = "") const;

    /**
     * Unregister a stream that was previously registered. After this call, no more events
     * can be recorded on the stream.
     */
    void unregister_stream(DeviceStreamId stream_id) const;

    /**
     * Returns the id of the given stream, or `std::nullopt` if it was not registered.
     */
    std::optional<DeviceStreamId> lookup_stream(GPUStreamId target) const;

    /**
     * Returns the id of the given stream, or `std::nullopt` if it was not registered.
     */
    DeviceStreamId lookup_or_register_stream(GPUStreamRef stream_ref) const;

    /**
     * Calls `unregister_stream` on all registered streams.
     */
    void shutdown() const;

    /**
     * Get the GPUStream associated with the given stream.
     */
    g_stream_t get(DeviceStreamId stream_id) const;

    /**
     * Get the GPUContext associated with the given stream.
     */
    g_context_t context(DeviceStreamId stream_id) const;

    /**
     * Get a `DeviceStream` wrapping the given stream.
     */
    DeviceStream stream(DeviceStreamId stream_id) const;

    /**
     * Check if the given stream id matches the given context id.
     */
    bool has_context(DeviceStreamId stream_id, GPUContextId context_id) const;

    /**
     * Record a new event on the given stream.
     */
    DeviceEvent record(DeviceStreamId stream_id) const;

    /**
     * Convenience method for the common pattern of waiting on dependencies, submitting work
     * onto a stream, and recording the resulting event. Waits for `deps`, calls
     * `fun(get(stream_id))` to submit work onto the stream, then records and returns the
     * resulting event.
     */
    template<typename F>
    DeviceEvent submit(DeviceStreamId stream_id, const DeviceEventSet& deps, F&& fun) const {
        wait_on_event(stream_id, deps);
        std::forward<F>(fun)(get(stream_id));
        return record(stream_id);
    }

    /**
     * Let the given stream wait until the given event completes.
     */
    void wait_on_event(DeviceStreamId stream_id, DeviceEvent event) const;

    /**
     * Let the given stream wait until all the given events completes.
     */
    void wait_on_event(DeviceStreamId stream_id, const DeviceEventSet& events) const;

    /**
     * Let the given stream wait until the given event completes.
     */
    void wait_on_event(g_stream_t stream, DeviceEvent event) const;

    /**
     * Let the given stream wait until all the given events completes.
     */
    void wait_on_event(g_stream_t stream, const DeviceEventSet& events) const;

    /**
     * Let the given stream wait until all work already enqueued on the CUDA legacy default
     * stream (i.e., stream `0`) in the same context completes. Used to synchronize with
     * work launched outside of KMM's own tracked streams.
     */
    void wait_on_default_stream(DeviceStreamId stream_id) const;

    /**
     * Check if the given stream is currently idle.
     */
    bool is_ready(DeviceStreamId stream_id) const;

    /**
     * Check if the given event has completed.
     */
    bool is_ready(DeviceEvent event) const;

    /**
     * Check if all the given events have completed.
     */
    bool is_ready(const DeviceEventSet& events) const;

    /**
     * Check if all events recorded on all streams have completed.
     */
    bool is_all_ready() const;

    /**
     * Check if the given event is latest known recorded event on its stream.
     */
    bool is_latest(DeviceEvent event) const;

    /**
     * Check if one of the given events is latest known recorded event on the given stream id.
     */
    bool is_latest_in(DeviceStreamId stream_id, const DeviceEventSet& deps) const;

    /**
     * Returns the latest recorded event on the given stream.
     */
    DeviceEvent latest_event(DeviceStreamId stream_id) const;

    /**
     * Block until the given stream becomes idle.
     */
    void synchronize(DeviceStreamId stream_id) const;

    /**
     * Block until the given event completes.
     */
    void synchronize(DeviceEvent event) const;

    /**
     * Block until the given events complete.
     */
    void synchronize(const DeviceEventSet& events) const;

    /**
     * Block until all events on all streams complete.
     */
    void synchronize_all() const;

    /**
     * Return a list of events that MUST have completed once the given stream completes.
     */
    DeviceEventSet snapshot(DeviceStreamId stream_id) const;

    /**
     * Attach a callback that fires when the given event completes.
     */
    void attach_callback(DeviceEvent event, NotifyHandle callback) const;

    /**
     * Poll the registered streams.
     */
    void make_progress() const;

    /**
     * Each stream keeps a pool of available events. Once an event completes, the free event
     * is added to this pool to be reused again later.
     *
     * This method clears all these pools and frees these events again.
     */
    void trim_event_pool(DeviceStreamId stream_id) const;

    /**
     * Calls `trim_event_pool` for all streams.
     */
    void trim_event_pool() const;

    /**
     * Check if the given event `a` precedes the given stream `b`. In other words, the
     * given event MUST complete before the stream becomes idle. This means that there exist
     * a dependency between the given event a and the stream b.
     *
     * Note: this is a hint. If `true`, then `a` MUST precede `b`. However, if `false, then
     * it might still be the case but `DeviceEventRegistry` has not witnessed the synchronization.
     */
    bool precedes(const DeviceEvent& a, const DeviceStreamId& b) const;

    /**
     * Shorthand that checks if `precedes(x, b)` for all x in a.
     */
    bool precedes(const DeviceEventSet& a, const DeviceStreamId& b) const;

    struct Impl;

  private:
    refcnt_ptr<Impl> m_impl;
};

KMM_REFCNT_TRAITS_FWD(DeviceEventRegistry::Impl)

}  // namespace kmm