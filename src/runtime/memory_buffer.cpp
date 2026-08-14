#include "fmt/chrono.h"
#include "spdlog/spdlog.h"

#include "kmm/runtime/memory_buffer.hpp"

namespace kmm {

Poll HostAccessControl::poll_pending_future() {
    if (pending_future.valid()) {
        if (pending_future.wait_for(std::chrono::seconds(0)) == std::future_status::timeout) {
            return Poll::Pending;
        }

        // will not block but does clear the future and handle any exceptions
        wait_pending_future();
    }

    return Poll::Ready;
}

void HostAccessControl::wait_pending_future() {
    if (pending_future.valid()) {
        auto before = std::chrono::system_clock::now();

        try {
            pending_future.get();
        } catch (...) {
            pending_future = {};
            is_valid = false;
            throw;
        }

        auto after = std::chrono::system_clock::now();
        auto duration = after - before;

        if (duration > std::chrono::milliseconds(1)) {
            spdlog::warn("waited for {} for host before to become available", duration);
        }
    }
}

bool MemoryBufferImpl::is_compatible(MemoryId memory_id, Access mode) noexcept {
    for (auto r = queue_head; r != nullptr; r = r->queue_next) {
        // two exclusive access are never allowed
        if (r->mode == Access::Exclusive || mode == Access::Exclusive) {
            return false;
        }

        // if one writes, then they must access the same memory
        if (r->mode != Access::ReadOnly || mode != Access::ReadOnly) {
            if (r->memory_id != memory_id) {
                return false;
            }
        }
    }

    // if we reach this point, then access has been granted
    return true;
}

bool MemoryBufferImpl::try_register_request(BufferQueueNode* req) noexcept {
    if (released) {
        return false;
    }

    if (!is_compatible(req->memory_id, req->mode)) {
        return false;
    }

    req->queue_prev = queue_tail;
    req->queue_next = nullptr;

    if (queue_tail != nullptr) {
        queue_tail->queue_next = req;
    } else {
        queue_head = req;
    }

    queue_tail = req;
    return true;
}

void MemoryBufferImpl::unregister_request(BufferQueueNode* req) noexcept {
    if (req->queue_prev != nullptr) {
        req->queue_prev->queue_next = req->queue_next;
    } else {
        queue_head = req->queue_next;
    }

    if (req->queue_next != nullptr) {
        req->queue_next->queue_prev = req->queue_prev;
    } else {
        queue_tail = req->queue_prev;
    }

    req->queue_prev = nullptr;
    req->queue_next = nullptr;
}

AllocResult MemoryBufferImpl::try_allocate_location(
    const DeviceStreamId& stream_hint,
    MemoryId dst_id
) {
    auto& dst_loc = location(dst_id);
    KMM_ASSERT(!dst_loc.is_allocated);

    MemoryId src_id = dst_id;
    bool has_peer = find_valid_location(dst_id, src_id);

    // A host operation migt still be active. Calls treat returning `false` as out of memory
    // failure. Instead, we fall back to a regular allocation and let the caller ensure that
    // the data is copied when the future completes.
    bool peer_ready = !src_id.is_host() || host_location.poll_pending_future() == Poll::Ready;

    if (has_peer && peer_ready
        && (dst_id.is_host() || src_id.is_host() || data->is_copy_supported(src_id, dst_id))) {
        auto& src_loc = location(src_id);
        DeviceEventSet events;
        AllocResult result = data->allocate_and_copy(
            src_id,
            dst_id,
            stream_hint,
            src_loc.retrieve_access(Access::SharedWrite),
            events
        );

        if (result == AllocResult::Success) {
            src_loc.record_access(Access::ReadOnly, events);
            dst_loc.mark_allocated_and_valid(events);
        }

        return result;
    }

    DeviceEventSet deps;
    AllocResult result = data->allocate(dst_id, stream_hint, deps);

    if (result == AllocResult::Success) {
        dst_loc.mark_allocated(std::move(deps));
    }

    return result;
}

bool MemoryBufferImpl::allocate_host(const DeviceStreamId& stream_hint) {
    spdlog::debug("allocate buffer {} in memory {}", name, MemoryId::host());

    if (try_allocate_location(stream_hint, MemoryId::host()) != AllocResult::Success) {
        throw std::runtime_error("could not allocate, out of host memory");
    }

    return true;
}

void MemoryBufferImpl::increment_host_users() noexcept {
    spdlog::trace(
        "buffer {}: host alloc_count {} -> {}",
        name,
        host_location.alloc_count,
        host_location.alloc_count + 1
    );
    host_location.alloc_count++;
}

void MemoryBufferImpl::decrement_host_users() noexcept {
    spdlog::trace(
        "buffer {}: host alloc_count {} -> {}",
        name,
        host_location.alloc_count,
        host_location.alloc_count - 1
    );
    KMM_ASSERT(host_location.alloc_count > 0);
    host_location.alloc_count--;
}

bool MemoryBufferImpl::deallocate_host(const DeviceStreamId& stream_hint) {
    auto& loc = host_location;
    KMM_ASSERT(loc.alloc_count == 0);

    if (!is_allocated(MemoryId::host())) {
        return false;
    }

    // This is the one place that actually frees the host allocation, so unlike
    // `invalidate_other_allocs`/`invalidate_all` (which may leave a stale future draining in
    // the background) it must force the location to finish.
    loc.wait_pending_future();

    spdlog::debug("deallocate buffer {} in memory {}", name, MemoryId::host());

    auto deps = loc.mark_deallocated();
    data->deallocate(MemoryId::host(), stream_hint, std::move(deps));
    return true;
}

AllocResult MemoryBufferImpl::try_allocate_device(const DeviceStreamId& stream_hint, DeviceId id) {
    spdlog::debug("allocate buffer {} in memory {}", name, MemoryId::device(id));
    return try_allocate_location(stream_hint, MemoryId::device(id));
}

bool MemoryBufferImpl::deallocate_device(
    const DeviceStreamId& stream_hint,
    DeviceId id,
    DeviceLRU& lru
) {
    auto& loc = device_locations[id.get()];
    KMM_ASSERT(loc.alloc_count == 0);

    if (!is_allocated(MemoryId::device(id))) {
        return false;
    }

    spdlog::debug("deallocate buffer {} in memory {}", name, MemoryId::device(id));

    auto deps = loc.mark_deallocated();
    data->deallocate(MemoryId::device(id), stream_hint, std::move(deps));
    lru.remove(&loc);

    return true;
}

void MemoryBufferImpl::increment_device_users(DeviceId id, DeviceLRU& lru) noexcept {
    auto& loc = device_locations[id.get()];
    spdlog::trace(
        "buffer {}: device {} alloc_count {} -> {}",
        name,
        id,
        loc.alloc_count,
        loc.alloc_count + 1
    );
    loc.alloc_count++;

    if (loc.alloc_count == 1) {
        lru.remove(&loc);
    }
}

void MemoryBufferImpl::decrement_device_users(DeviceId id, DeviceLRU& lru) noexcept {
    auto& loc = device_locations[id.get()];
    KMM_ASSERT(loc.alloc_count > 0);
    spdlog::trace(
        "buffer {}: device {} alloc_count {} -> {}",
        name,
        id,
        loc.alloc_count,
        loc.alloc_count - 1
    );
    loc.alloc_count--;

    if (loc.alloc_count == 0 && evictable) {
        lru.insert(&loc);
    }
}

void MemoryBufferImpl::evict_device(
    const DeviceStreamId& stream_hint,
    DeviceId memory_id,
    DeviceLRU& lru
) {
    auto& loc = device_locations[memory_id.get()];
    KMM_ASSERT(is_allocated(MemoryId::device(memory_id)));

    auto other_id = MemoryId::host();

    // if this entry is valid and there are no other valid entries, then we must evict to host
    if (loc.is_valid && !find_valid_location(MemoryId::device(memory_id), other_id)) {
        // `allocate_host` finds this device (still valid, not yet deallocated) as its copy
        // source and copies from it directly -- src is a device, so this can never be Pending.
        allocate_host(stream_hint);
    }

    deallocate_device(stream_hint, memory_id, lru);
}

Poll MemoryBufferImpl::ensure_alloc_valid(const DeviceStreamId& stream_hint, MemoryId memory_id) {
    auto& loc = location(memory_id);

    // If we are accessing the host, we must first check if the future on the host is ready.
    // This is needed for both the case that is is_valid==true and is_valid==false.
    if (memory_id.is_host() && host_location.poll_pending_future() == Poll::Pending) {
        return Poll::Pending;
    }

    // if already valid, we just exit. Even for the host, there can not be a future pending.
    if (loc.is_valid) {
        return Poll::Ready;
    }

    MemoryId peer_id = memory_id;
    bool has_valid_peer = find_valid_location(memory_id, peer_id);

    if (!has_valid_peer) {
        //
        const auto& deps = location(memory_id).retrieve_access(Access::Exclusive);

        // This now becomes the home memory
        if (!home_memory_id.has_value()) {
            home_memory_id = memory_id;
        }

        spdlog::debug("initializing buffer {} on {}", name, memory_id);

        if (memory_id.is_host()) {
            host_location.pending_future = data->initialize_host(deps);
            loc.mark_valid(DeviceEvent::null());
            return host_location.poll_pending_future();
        } else {
            auto event = data->initialize_device(memory_id.as_device(), stream_hint, deps);
            loc.mark_valid(event);
            return Poll::Ready;
        }
    } else if (memory_id.is_host() || peer_id.is_host() || data->is_copy_supported(peer_id, memory_id)) {
        // copy D2H or H2D or D2D (if possible)
        return poll_copy(stream_hint, peer_id, memory_id);
    } else {
        // copy D2H -> H2D: `allocate_host` finds `peer_id` (or another valid location) as its
        // copy source and performs the D2H leg itself; only the H2D leg remains here.
        allocate_host(stream_hint);
        do_copy(stream_hint, MemoryId::host(), memory_id);
        return Poll::Ready;
    }
}

void MemoryBufferImpl::invalidate_other_allocs(MemoryId memory_id) {
    DeviceEventSet deps;

    if (memory_id.is_host()) {
        // Invalidate all  device entries
        for (size_t i = 0; i < MAX_DEVICES; i++) {
            auto& peer_entry = device_locations[i];
            deps.insert(peer_entry.retrieve_access(Access::ReadOnly));
            peer_entry.is_valid = false;
        }
    } else {
        // Invalidate host if necessary. We deliberately do NOT wait for a still-running host
        // future here: the underlying allocation stays put (only `deallocate_host` frees it,
        // and it force-drains any in-flight future first, so there's no use-after-free risk),
        // and the only other hazard -- a later host fill overwriting `pending_future` while
        // this one is still outstanding -- is guarded non-blockingly in `ensure_alloc_valid`.
        deps.insert(host_location.retrieve_access(Access::ReadOnly));
        host_location.is_valid = false;

        // Invalidate all _other_ device entries
        for (size_t i = 0; i < MAX_DEVICES; i++) {
            if (memory_id == MemoryId::device(DeviceId(i))) {
                continue;
            }

            auto& peer_entry = device_locations[i];
            deps.insert(peer_entry.retrieve_access(Access::ReadOnly));
            peer_entry.is_valid = false;
        }
    }

    location(memory_id).record_access(Access::Exclusive, deps);
}

DeviceEventSet MemoryBufferImpl::invalidate_all() {
    if (queue_head != nullptr) {
        throw std::runtime_error("failed to invalidate buffer as access is locked by a request");
    }

    DeviceEventSet deps;

    // See `invalidate_other_allocs`: no need to wait out an in-flight host future here --
    // the allocation isn't freed by this call, and `ensure_alloc_valid` non-blockingly drains
    // any leftover future before a new one could overwrite it.
    deps.insert(host_location.retrieve_access(Access::ReadOnly));
    host_location.is_valid = false;

    for (size_t i = 0; i < MAX_DEVICES; i++) {
        auto& peer_entry = device_locations[i];
        deps.insert(peer_entry.retrieve_access(Access::ReadOnly));
        peer_entry.is_valid = false;
    }

    return deps;
}

Poll MemoryBufferImpl::before_access(
    const DeviceStreamId& stream_hint,
    MemoryId memory_id,
    Access mode,
    DeviceEventSet& deps_out
) {
    // 1) ensure that the allocation contains valid data
    if (ensure_alloc_valid(stream_hint, memory_id) == Poll::Pending) {
        return Poll::Pending;
    }

    // 2) if we are going to write, we must invalidate all the others
    if (mode != Access::ReadOnly) {
        invalidate_other_allocs(memory_id);
    }

    // wait for:
    // Exclusive waits for Exclusive, SharedWrite, ReadOnly
    // ReadOnly wait for Exclusive, SharedWrite
    // SharedWrite wait for Exclusive
    Access waiting_mode;
    if (mode == Access::Exclusive) {
        waiting_mode = Access::ReadOnly;
    } else if (mode == Access::ReadOnly) {
        waiting_mode = Access::SharedWrite;
    } else {
        waiting_mode = Access::Exclusive;
    }

    // 3) collect the dependencies needed before the buffer can be accessed
    auto& loc = location(memory_id);
    const auto& deps = loc.retrieve_access(waiting_mode);
    data->hint_access(memory_id, stream_hint, deps);
    deps_out.insert(deps);

    return Poll::Ready;
}

void MemoryBufferImpl::after_access(MemoryId memory_id, Access mode, const DeviceEventSet& deps) {
    location(memory_id).record_access(mode, deps);
}

Poll MemoryBufferImpl::poll_copy(
    const DeviceStreamId& stream_hint,
    MemoryId src_id,
    MemoryId dst_id
) {
    // if this involves the host, we must first wait until the associated future completes. If
    // the future is still active, then there may still be threads actively reading/writing
    // to the host memory and we must wait until they complete.
    if ((src_id.is_host() || dst_id.is_host())
        && host_location.poll_pending_future() == Poll::Pending) {
        return Poll::Pending;
    }

    do_copy(stream_hint, src_id, dst_id);
    return Poll::Ready;
}

void MemoryBufferImpl::do_copy(
    const DeviceStreamId& stream_hint,
    MemoryId src_id,
    MemoryId dst_id
) {
    auto& src_alloc = location(src_id);
    auto& dst_alloc = location(dst_id);

    KMM_ASSERT(src_alloc.is_allocated && dst_alloc.is_allocated);
    KMM_ASSERT(src_alloc.is_valid && !dst_alloc.is_valid);

    // if this involves the host, then there cannot be any futures active on the host memory.
    KMM_ASSERT((!src_id.is_host() && !dst_id.is_host()) || !host_location.pending_future.valid());

    DeviceEventSet deps;
    deps.insert(src_alloc.retrieve_access(Access::SharedWrite));
    deps.insert(dst_alloc.retrieve_access(Access::ReadOnly));

    spdlog::debug("launch copy for buffer {} from {} to {} (deps: {})", name, src_id, dst_id, deps);
    data->copy(src_id, dst_id, stream_hint, deps, deps);

    src_alloc.record_access(Access::ReadOnly, deps);
    dst_alloc.mark_valid(deps);
}

BufferAccessor MemoryBufferImpl::accessor(MemoryId memory_id, Access mode) {
    return BufferAccessor {
        memory_id,
        size_in_bytes,
        mode != Access::ReadOnly,
        data->address(memory_id)};
}

}  // namespace kmm