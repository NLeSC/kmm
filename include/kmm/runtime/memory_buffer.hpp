#pragma once

#include <optional>

#include "kmm/runtime/data_interface.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

struct BufferQueueNode {
    BufferQueueNode(MemoryId memory_id, Access mode, NotifyHandle callback) :
        memory_id(memory_id),
        mode(mode) {}

    // Intrusive doubly-linked list used by `MemoryBufferImpl`'s granted/pending queue.
    BufferQueueNode* queue_prev = nullptr;
    BufferQueueNode* queue_next = nullptr;
    MemoryId memory_id;
    Access mode;
};

struct AccessControl {
    // epoch_events: events created by Epoch access. All future requests must wait for
    //               all epoch events to finish before they can access the data.
    // write_events: events created by Epoch or Write access. Future reads must
    //               wait for all write events before they may safely observe the data.
    // read_events: events created by Epoch or Write or Read access. Future writes
    //              must wait for all read events before they may safely overwrite the data.
    //
    // The ordering is always:
    // - if event in epoch_events => event in write_events
    // - if event in write_events => event in read_events
    DeviceEventSet epoch_events;
    DeviceEventSet write_events;
    DeviceEventSet read_events;

    // Whether this location holds (or will hold, once any producer finishes) semantically
    // correct data. For `HostAccessControl`, this is independent of `pending_future`: a
    // location can be `is_valid == true` while a fill is still in flight (readers must drain
    // `pending_future` first), and it can also be `is_valid == false` while a *stale* future
    // from before an invalidation is still draining in the background (see
    // `MemoryBufferImpl::invalidate_other_allocs`/`invalidate_all`).
    bool is_valid = false;
    bool is_allocated = false;
    size_t alloc_count = 0;

    // Records an access of the given `mode`, maintaining the epoch/write/read
    // nesting invariant documented above.
    void record_access(Access mode, const DeviceEventSet& deps) noexcept {
        if (mode == Access::Exclusive) {
            epoch_events.insert(deps);
        }

        if (mode != Access::ReadOnly) {
            write_events.insert(deps);
        }

        read_events.insert(deps);
    }

    const DeviceEventSet& retrieve_access(Access mode) noexcept {
        switch (mode) {
            case Access::ReadOnly:
                return read_events;
            case Access::SharedWrite:
                return write_events;
            case Access::Exclusive:
                return epoch_events;
            default:
                KMM_PANIC("invalid state");
        }
    }

    void mark_allocated(DeviceEventSet deps) {
        KMM_ASSERT(alloc_count == 0);
        KMM_ASSERT(!is_allocated);
        is_allocated = true;
        is_valid = false;
        write_events = deps;
        read_events = deps;
        epoch_events = std::move(deps);
    }

    void mark_valid(const DeviceEventSet& events) noexcept {
        epoch_events.insert(events);
        write_events.insert(events);
        read_events.insert(events);
        is_valid = true;
    }

    void mark_allocated_and_valid(const DeviceEventSet& events) noexcept {
        KMM_ASSERT(alloc_count == 0);
        KMM_ASSERT(!is_allocated);
        mark_valid(events);
        is_allocated = true;
    }

    // Returns the dependencies the caller must wait on before actually freeing the
    // underlying allocation (via `DataInterface`), then clears all state.
    DeviceEventSet mark_deallocated() noexcept {
        DeviceEventSet deps = retrieve_access(Access::ReadOnly);
        KMM_ASSERT(alloc_count == 0);
        is_allocated = false;
        is_valid = false;
        epoch_events.clear();
        write_events.clear();
        read_events.clear();
        return deps;
    }
};

struct HostAccessControl: AccessControl {
    // Tracks an in-flight async host producer (currently only `DataInterface::initialize_host`).
    // Mostly independent of `is_valid`: the caller decides what a drained future implies (e.g.
    // `ensure_alloc_valid` marks the location valid once it starts a fresh producer; a stale
    // future left over from an invalidation is just discarded once drained). The one exception
    // is failure: if the producer throws, `is_valid` is forced back to `false` here, since a
    // failed producer never actually produced valid data, regardless of what the caller assumed
    // when it started the future.
    std::future<void> pending_future;

    Poll poll_pending_future();
    void wait_pending_future();
};

struct DeviceAccessControl: AccessControl {
    // Number of currently-granted requests holding this exact location. Only
    // locations with `alloc_count == 0` may sit in the LRU list / be evicted.
    bool in_lru = false;
    DeviceAccessControl* lru_prev = nullptr;
    DeviceAccessControl* lru_next = nullptr;
};

struct DeviceLRU {
    DeviceAccessControl* head = nullptr;
    DeviceAccessControl* tail = nullptr;

    DeviceAccessControl* least_recently_used() const noexcept {
        return head;
    }

    void insert(DeviceAccessControl* loc) noexcept {
        KMM_ASSERT(!loc->in_lru);
        loc->lru_prev = tail;
        loc->lru_next = nullptr;

        if (tail != nullptr) {
            tail->lru_next = loc;
        } else {
            head = loc;
        }

        tail = loc;
        loc->in_lru = true;
    }

    void remove(DeviceAccessControl* loc) noexcept {
        if (!loc->in_lru) {
            return;
        }

        if (loc->lru_prev != nullptr) {
            loc->lru_prev->lru_next = loc->lru_next;
        } else {
            head = loc->lru_next;
        }

        if (loc->lru_next != nullptr) {
            loc->lru_next->lru_prev = loc->lru_prev;
        } else {
            tail = loc->lru_prev;
        }

        loc->lru_prev = nullptr;
        loc->lru_next = nullptr;
        loc->in_lru = false;
    }
};

struct MemoryBufferImpl: reference_count<MemoryBufferImpl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemoryBufferImpl)
  public:
    const size_t size_in_bytes;

    MemoryBufferImpl(
        std::string name,
        bool evictable,
        std::unique_ptr<DataInterface> data,
        std::optional<MemoryId> home_memory_id = {}
    ) :
        size_in_bytes(data->size_in_bytes()),
        name(std::move(name)),
        evictable(evictable),
        data(std::move(data)),
        home_memory_id(home_memory_id) {}

    bool is_compatible(MemoryId memory_id, Access mode) noexcept;
    bool try_register_request(BufferQueueNode* req) noexcept;
    void unregister_request(BufferQueueNode* req) noexcept;

    AccessControl& location(MemoryId id) noexcept {
        if (id.is_host()) {
            return host_location;
        } else {
            return device_locations[id.as_device().get()];
        }
    }

    const AccessControl& location(MemoryId id) const noexcept {
        if (id.is_host()) {
            return host_location;
        } else {
            return device_locations[id.as_device().get()];
        }
    }

    // A host location that's still `is_valid` while its producer is in flight counts as
    // "found" here too: this only picks a copy *source*, and the eventual consumer
    // (`poll_copy`/`do_copy`) waits out `pending_future` before actually reading it. Treating
    // an in-flight producer as absent instead would make callers re-produce the data from
    // scratch (redundant, and wrong for a non-idempotent producer) instead of reusing what's
    // already in flight.
    bool find_valid_location(MemoryId exclude, MemoryId& out) const noexcept {
        // Prefer the buffer's home location, if it has one and it's still valid.
        if (home_memory_id.has_value()) {
            if (*home_memory_id != exclude && location(*home_memory_id).is_valid) {
                out = *home_memory_id;
                return true;
            }
        }

        if (host_location.is_valid && MemoryId::host() != exclude) {
            out = MemoryId::host();
            return true;
        }

        for (size_t id = 0; id < MAX_DEVICES; id++) {
            if (device_locations[id].is_valid && MemoryId::device(DeviceId(id)) != exclude) {
                out = MemoryId::device(DeviceId(id));
                return true;
            }
        }

        return false;
    }

    MemoryId find_preferred_location(MemoryId fallback) {
        if (is_valid(fallback)) {
            return fallback;
        }

        if (find_valid_location(fallback, fallback)) {
            return fallback;
        }

        if (home_memory_id.has_value()) {
            return *home_memory_id;
        }

        return fallback;
    }

    // Strict: `true` if the data has a producer (either data is available or will be available).
    bool is_valid(MemoryId id) {
        return location(id).is_valid;
    }

    bool is_allocated(MemoryId id) {
        return location(id).is_allocated;
    }

    AllocResult try_allocate_location(const DeviceStreamId& stream_hint, MemoryId dst_id);

    bool allocate_host(const DeviceStreamId& stream_hint);
    bool deallocate_host(const DeviceStreamId& stream_hint);
    void increment_host_users() noexcept;
    void decrement_host_users() noexcept;

    AllocResult try_allocate_device(const DeviceStreamId& stream_hint, DeviceId id);
    bool deallocate_device(const DeviceStreamId& stream_hint, DeviceId id, DeviceLRU& lru);
    void increment_device_users(DeviceId id, DeviceLRU& lru) noexcept;
    void decrement_device_users(DeviceId id, DeviceLRU& lru) noexcept;

    void evict_device(const DeviceStreamId& stream_hint, DeviceId id, DeviceLRU& lru);
    Poll ensure_alloc_valid(const DeviceStreamId& stream_hint, MemoryId memory_id);
    void invalidate_other_allocs(MemoryId memory_id);
    DeviceEventSet invalidate_all();

    // Called once a request's location has been granted, right before the
    // caller is allowed to actually read/write through it.
    Poll before_access(const DeviceStreamId& stream_hint, MemoryId memory_id, Access mode);

    // Called right after the caller is done reading/writing through a
    // granted location, recording the resulting dependencies.
    void after_access(MemoryId memory_id, Access mode, const DeviceEventSet& deps);

    // Returns `Pending` (without touching any state) if `src_id` is host and its data is
    // still being produced by an in-flight `pending_future`.
    Poll poll_copy(const DeviceStreamId& stream_hint, MemoryId src_id, MemoryId dst_id);
    void do_copy(const DeviceStreamId& stream_hint, MemoryId src_id, MemoryId dst_id);

    // Returns the accessor granting access to this buffer (once `before_access` has returned
    // `Ready`), and inserts into `deps_out` the events that must complete before it is safe to
    // read/write through it. Reads live off the location's current epoch/write/read events, so
    // this may safely be called after `before_access` rather than only from within it.
    BufferAccessor access(MemoryId memory_id, Access mode, DeviceEventSet& deps_out);

    const std::string name;

    // If false, this buffer's device locations are never inserted into the LRU.
    const bool evictable;

    std::unique_ptr<DataInterface> data;

    HostAccessControl host_location;
    DeviceAccessControl device_locations[MAX_DEVICES];

    BufferQueueNode* queue_head = nullptr;
    BufferQueueNode* queue_tail = nullptr;

    // The location of the first access ever granted to this buffer. Set once, by
    // `before_access`, and never changed afterwards.
    std::optional<MemoryId> home_memory_id;

    // Set by `MemoryManager::release_buffer` before it tears down allocations,
    // to guard against the buffer being used in a new transaction while that
    // teardown is in progress.
    bool released = false;
};

}  // namespace kmm