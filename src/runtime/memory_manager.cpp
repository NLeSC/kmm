#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "kmm/core/panic.hpp"
#include "kmm/runtime/memory_buffer.hpp"
#include "kmm/runtime/memory_manager.hpp"

namespace kmm {

struct MemoryTransactionImpl: reference_count<MemoryTransactionImpl> {
    explicit MemoryTransactionImpl(MemoryTransaction parent) : parent(std::move(parent)) {}
    MemoryTransaction parent;  // may be null
};

struct DeviceQueueNode {
    DeviceQueueNode(NotifyHandle callback) : callback(std::move(callback)) {}

    NotifyHandle callback;

    // Intrusive doubly-linked list used by `DeviceState`: a single list holding
    // both the allocated prefix and the waiting suffix, split by `req_first_pending`.
    DeviceQueueNode* device_prev = nullptr;
    DeviceQueueNode* device_next = nullptr;
    MemoryTransaction parent;
};

struct MemoryRequestImpl: BufferQueueNode, DeviceQueueNode, reference_count<MemoryRequestImpl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemoryRequestImpl)

  public:
    MemoryRequestImpl(
        refcnt_ptr<MemoryBufferImpl> buffer,
        MemoryId memory_id,
        Access mode,
        NotifyHandle callback
    ) :
        BufferQueueNode(memory_id, mode, callback),
        DeviceQueueNode(callback),
        buffer(std::move(buffer)) {}

    enum struct State { Unqueued, WaitingForAllocation, Granted, Ready, Released };

    refcnt_ptr<MemoryBufferImpl> buffer;
    State state = State::Unqueued;
    void* pointer = nullptr;
};

// Recovers the `MemoryBufferImpl` owning `device_locations[id.get()]` from a
// pointer to that location, using the fact that it lives at a fixed offset
// inside its owner (the LRU list only ever holds device locations, never
// `host_location`, so `id` is enough to identify which slot `loc` is).
static MemoryBufferImpl* owner_of_device_location(DeviceAccessControl* loc, DeviceId id) noexcept {
    DeviceAccessControl* first = loc - id.get();

    // `MemoryBufferImpl` derives from `reference_count`, so it is not standard-layout and
    // `offsetof` is technically only conditionally-supported here. The offset is still valid in
    // practice (single, non-virtual inheritance), so silence the warning rather than the check.
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
    auto offset = offsetof(MemoryBufferImpl, device_locations);
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
#endif

    return reinterpret_cast<MemoryBufferImpl*>(reinterpret_cast<char*>(first) - offset);
}

// Host-memory counterpart of `DeviceState`: there is only ever one host
// location per buffer, allocation never blocks or needs to evict to make
// room, so there is no LRU and no request queue here, just byte accounting.
struct HostState {
    // Request-scoped acquire: bundles allocation (if needed) with the usage-count
    // bump. Its counterpart `release_for_request` only undoes the usage-count
    // half — deallocation is buffer-scoped, not request-scoped, see `deallocate`.
    void acquire_for_request(DeviceStream stream_hint, MemoryBufferImpl* buf) {
        if (!buf->is_allocated(MemoryId::host())) {
            buf->allocate_host(stream_hint);
            bytes_allocated += buf->size_in_bytes;
        }

        buf->increment_host_users();
    }

    void release_for_request(MemoryBufferImpl* buf) noexcept {
        buf->decrement_host_users();
    }

    void deallocate(DeviceStream stream_hint, MemoryBufferImpl* buf) noexcept {
        if (buf->deallocate_host(stream_hint)) {
            bytes_allocated -= buf->size_in_bytes;
        }
    }

    // Total number of bytes currently allocated on the host across all buffers.
    size_t bytes_allocated = 0;
};

struct DeviceState {
    DeviceState(DeviceId id) noexcept : memory_id(id) {}

    void register_request(DeviceQueueNode* req) noexcept {
        req->device_prev = req_tail;
        req->device_next = nullptr;

        if (req_tail != nullptr) {
            req_tail->device_next = req;
        } else {
            req_head = req;
        }

        req_tail = req;

        if (req_first_pending == nullptr) {
            req_first_pending = req;
        } else {
            // This request joins behind an already-waiting `req_first_pending`.
            // Its transaction's ancestor chain may reach a transaction that
            // `is_out_of_memory` previously judged not blocked (e.g. if this
            // request's transaction is a descendant of one already holding
            // memory here), which would flip that earlier conclusion. Wake
            // `req_first_pending` so it rechecks instead of waiting forever
            // on a now-stale answer.
            req_first_pending->callback.notify();
        }
    }

    void unregister_request(DeviceQueueNode* req) noexcept {
        if (req->device_prev != nullptr) {
            req->device_prev->device_next = req->device_next;
        } else {
            req_head = req->device_next;
        }

        if (req->device_next != nullptr) {
            req->device_next->device_prev = req->device_prev;
        } else {
            req_tail = req->device_prev;
        }

        if (req_first_pending == req) {
            req_first_pending = req->device_next;

            // another request became the head of the pending queue, notify it
            if (req_first_pending != nullptr) {
                req_first_pending->callback.notify();
            }
        }

        req->callback.clear();
        req->device_prev = nullptr;
        req->device_next = nullptr;
    }

    // Counterpart of `try_acquire_for_request` for release: only undoes the
    // usage-count half of acquire, since deallocation is buffer-scoped, not
    // request-scoped (see `deallocate`, only called from buffer release).
    void release_for_request(MemoryBufferImpl* buf) noexcept {
        buf->decrement_device_users(memory_id, lru);

        // if somebody is waiting, wake them up! We must always wake up the pending request,
        // even if we do not add anything to the LRU, since the request needs to check again
        // if the may be out of memory now.
        if (req_first_pending != nullptr) {
            req_first_pending->callback.notify();
        }
    }

    bool try_allocate(DeviceStream stream_hint, MemoryBufferImpl* buf) {
        if (!buf->is_allocated(MemoryId::device(memory_id))) {
            if (!buf->try_allocate_device(stream_hint, memory_id)) {
                return false;
            }

            bytes_allocated += buf->size_in_bytes;
        }

        buf->increment_device_users(memory_id, lru);
        return true;
    }

    void deallocate(DeviceStream stream_hint, MemoryBufferImpl* buf) noexcept {
        if (buf->is_allocated(MemoryId::device(memory_id))) {
            buf->deallocate_device(stream_hint, memory_id, lru);
            bytes_allocated -= buf->size_in_bytes;
        }
    }

    MemoryBufferImpl* select_evict_victim() {
        // First, find a buffer which has a different home location than memory_id
        for (auto* node = lru.least_recently_used(); node != nullptr; node = node->lru_next) {
            // This is done in a very hacky way by using some pointer trickery.
            MemoryBufferImpl* owner = owner_of_device_location(node, memory_id);

            if (owner->home_memory_id != MemoryId::device(memory_id)) {
                return owner;
            }
        }

        // Second, just select the most recently used
        if (auto* victim = lru.least_recently_used()) {
            return owner_of_device_location(victim, memory_id);
        }

        return nullptr;
    }

    bool try_evict_one(DeviceStream stream_hint) {
        if (auto* victim = select_evict_victim()) {
            return try_evict_buffer(stream_hint, victim);
        } else {
            return false;
        }
    }

    bool try_evict_buffer(DeviceStream stream_hint, MemoryBufferImpl* buf) {
        auto& loc = buf->device_locations[memory_id.get()];

        if (!loc.in_lru) {
            return false;
        }

        buf->evict_device(stream_hint, memory_id, lru);
        bytes_allocated -= buf->size_in_bytes;
        return true;
    }

    // Defined below, once `MemoryRequestImpl` is a complete type.
    bool try_acquire_for_request(DeviceStream stream_hint, MemoryRequestImpl* req);
    bool is_out_of_memory(MemoryRequestImpl* req);

    DeviceId memory_id;
    DeviceLRU lru;

    // Total number of bytes currently allocated on this device across all buffers
    // (i.e. the sum of `nbytes` of every buffer with an allocated `Location` here).
    size_t bytes_allocated = 0;

    /// This is a linked list of requests wanting to allocate memory.
    /// - req_head to req_first_pending: all requests that have been assigned memory
    /// - req_first_pending - req_tail: all requests that are still waiting for memory
    DeviceQueueNode* req_head = nullptr;
    DeviceQueueNode* req_first_pending = nullptr;
    DeviceQueueNode* req_tail = nullptr;
};

bool DeviceState::try_acquire_for_request(DeviceStream stream_hint, MemoryRequestImpl* req) {
    KMM_ASSERT(req_first_pending != nullptr);

    // only the first pending request may allocate
    if (req_first_pending != req) {
        return false;
    }

    auto* buf = req->buffer.get();

    // if it is already allocated, then we are done
    while (true) {
        if (try_allocate(stream_hint, buf)) {
            req->callback.clear();
            req_first_pending = req->device_next;

            // notify the next request that it can now attempt allocation
            if (req_first_pending != nullptr) {
                req_first_pending->callback.notify();
            }

            return true;
        }

        // Out of memory: reclaim the least-recently-used eligible location for
        // this device and retry. If successful, retry
        if (try_evict_one(stream_hint)) {
            continue;
        }

        if (is_out_of_memory(req)) {
            throw std::runtime_error("out of memory");
        }

        // Everything failed. return false
        return false;
    }
}

bool DeviceState::is_out_of_memory(MemoryRequestImpl* req) {
    KMM_ASSERT(req_first_pending == req);

    ankerl::unordered_dense::set<MemoryTransactionImpl*> not_blocked;

    // Transactions currently holding memory are inserted into not_blocked
    for (auto* it = req_head; it != req_first_pending; it = it->device_next) {
        for (auto* t = it->parent.get(); t != nullptr; t = t->parent.get()) {
            not_blocked.insert(t);
        }
    }

    // Transactions waiting for more memory are assume to be blocked and are thus removed
    // from not_blocked. This gives a list of transactions that have memory assigned and are
    // not currently waiting for more memory.
    for (auto* it = req_first_pending; it != nullptr; it = it->device_next) {
        for (auto* t = it->parent.get(); t != nullptr; t = t->parent.get()) {
            not_blocked.erase(t);
        }
    }

    // Whatever remains is a transaction that holds memory here and isn't
    // itself blocked waiting for more, i.e., it can complete on its own.
    if (!not_blocked.empty()) {
        return false;
    }

    // TODO: log warning
    return true;
}

struct MemoryManager::Impl {
    KMM_NOT_COPYABLE_OR_MOVABLE(Impl)

  public:
    Impl() : Impl(std::make_index_sequence<MAX_DEVICES>()) {}

  private:
    template<size_t... Is>
    Impl(std::index_sequence<Is...>) : device_states {DeviceState(DeviceId(Is))...} {}

  public:
    DeviceState& device(DeviceId id) {
        return device_states[id.get()];
    }

    HostState& host() {
        return host_state;
    }

    // Frees any device location that's sitting idle in its LRU while marked
    // invalid: it holds no useful data, so there is no reason to wait for dealloc.
    void reclaim_invalidated(DeviceStream stream_hint, MemoryBufferImpl* buf) {
        for (size_t i = 0; i < MAX_DEVICES; i++) {
            auto& loc = buf->device_locations[i];

            if (loc.in_lru && !loc.is_valid) {
                device(DeviceId(i)).deallocate(stream_hint, buf);
            }
        }
    }

    HostState host_state;
    DeviceState device_states[MAX_DEVICES];
};

MemoryManager::MemoryManager() : m_impl() {}

MemoryManager::~MemoryManager() {
    KMM_PANIC("todo: cleanup");
}

MemoryBuffer MemoryManager::create_buffer(
    std::unique_ptr<DataInterface> data,
    std::string name,
    bool evictable
) {
    return MemoryBuffer(make_refcnt<MemoryBufferImpl>(std::move(name), evictable, std::move(data)));
}

void MemoryManager::release_buffer(MemoryBuffer buffer) {
    auto* buf = buffer.get();

    // check if already released
    if (buf->released) {
        return;
    }

    buf->released = true;
    KMM_ASSERT(buf->queue_head == nullptr);
    DeviceStream stream_hint = DeviceStream {};

    m_impl->host().deallocate(stream_hint, buf);

    for (size_t id = 0; id < MAX_DEVICES; id++) {
        m_impl->device(DeviceId(id)).deallocate(stream_hint, buf);
    }
}

MemoryTransaction MemoryManager::create_transaction(MemoryTransaction parent) {
    return MemoryTransaction(make_refcnt<MemoryTransactionImpl>(std::move(parent)));
}

MemoryRequest MemoryManager::create_request(
    const MemoryBuffer& buffer,
    MemoryId memory_id,
    Access mode,
    MemoryTransaction parent,
    NotifyHandle callback
) {
    auto req = make_refcnt<MemoryRequestImpl>(
        refcnt_ptr<MemoryBufferImpl>(buffer.get(), true),
        memory_id,
        mode,
        std::move(callback)
    );

    // register with buffer
    if (req->buffer->try_register_request(req.get())) {
        // set state to waiting for allocation
        req->state = MemoryRequestImpl::State::WaitingForAllocation;
        req->parent = parent;

        // register with memory arbiter
        if (req->memory_id.is_device()) {
            m_impl->device(req->memory_id.as_device()).register_request(req.get());
        }
    } else {
        throw std::runtime_error("failed to lock buffer for access");
    }

    return req;
}

Poll MemoryManager::poll_request(
    DeviceStream stream_hint,
    const MemoryRequest& request,
    DeviceEventSet& deps_out
) {
    MemoryManager::Impl& mgr = *m_impl;
    MemoryRequestImpl* req = request.get();
    auto* buf = req->buffer.get();
    auto memory_id = req->memory_id;

    if (req->state == MemoryRequestImpl::State::Unqueued) {
        throw std::runtime_error("cannot poll memory request not registered with transaction");
    }

    if (req->state == MemoryRequestImpl::State::WaitingForAllocation) {
        if (memory_id.is_host()) {
            mgr.host().acquire_for_request(stream_hint, buf);
        } else {
            if (!mgr.device(memory_id.as_device()).try_acquire_for_request(stream_hint, req)) {
                return Poll::Pending;
            }
        }

        req->state = MemoryRequestImpl::State::Granted;
    }

    if (req->state == MemoryRequestImpl::State::Granted) {
        if (buf->before_access(stream_hint, memory_id, req->mode, deps_out) == Poll::Pending) {
            return Poll::Pending;
        }

        mgr.reclaim_invalidated(stream_hint, buf);
        req->state = MemoryRequestImpl::State::Ready;
    }

    KMM_ASSERT(req->state == MemoryRequestImpl::State::Ready);
    return Poll::Ready;
}

BufferAccessor MemoryManager::access_request(const MemoryRequest& request) {
    auto* req = request.get();
    auto* buf = req->buffer.get();

    KMM_ASSERT(req->state == MemoryRequestImpl::State::Ready);
    return buf->accessor(req->memory_id, req->mode);
}

void MemoryManager::release_request(MemoryRequest request, const DeviceEventSet& deps) {
    MemoryManager::Impl& mgr = *m_impl;
    auto* req = request.get();
    auto* buf = request->buffer.get();
    auto memory_id = request->memory_id;

    if (req->state == MemoryRequestImpl::State::Ready) {
        buf->after_access(memory_id, req->mode, deps);
        req->state = MemoryRequestImpl::State::Granted;
    }

    if (req->state == MemoryRequestImpl::State::Granted) {
        if (memory_id.is_device()) {
            mgr.device(memory_id.as_device()).release_for_request(buf);
        } else {
            mgr.host().release_for_request(buf);
        }

        req->state = MemoryRequestImpl::State::WaitingForAllocation;
    }

    if (req->state == MemoryRequestImpl::State::WaitingForAllocation) {
        if (memory_id.is_device()) {
            mgr.device(memory_id.as_device()).unregister_request(req);
        }

        buf->unregister_request(req);
        req->state = MemoryRequestImpl::State::Unqueued;
    }

    if (req->state == MemoryRequestImpl::State::Unqueued) {
        // Never activated: not present in any queue, nothing to unwind.
        req->state = MemoryRequestImpl::State::Released;
    }

    KMM_ASSERT(req->state == MemoryRequestImpl::State::Released);
}

void MemoryManager::prefetch_buffer(const MemoryBuffer& buffer, MemoryId memory_id, Access mode) {
    DeviceStream stream_hint = {};
    MemoryRequest req;

    // A single poll attempt: this is only a hint, so if the buffer cannot be
    // allocated/granted right away (contended, or out of memory), give up
    // instead of waiting like a real access would. `release_request` cleanly
    // unwinds whatever partial progress was made (e.g. allocated but not yet
    // granted), same as it does for any request that never reaches `Ready`.
    DeviceEventSet deps;

    try {
        req = create_request(buffer, memory_id, mode);

        if (poll_request(stream_hint, req, deps) == Poll::Ready) {
            // Say hi!
            access_request(req);
        }
    } catch (const std::exception&) {
        // e.g. out of memory, drop the hint.
    }

    if (req != nullptr) {
        release_request(req, deps);
    }
}

void MemoryManager::try_evict_buffer(const MemoryBuffer& buffer, MemoryId memory_id) {
    DeviceStream stream_hint;

    if (!buffer->is_allocated(memory_id)) {
        return;
    }

    if (memory_id.is_device()) {
        m_impl->device(memory_id.as_device()).try_evict_buffer(stream_hint, buffer.get());
        return;
    }

    // Host memory has no further fallback location: unlike a device eviction,
    // which can always fall back to copying into host memory first, deallocating
    // the host location while it holds the buffer's only valid copy would destroy
    // the data outright. Bail instead of doing that.
    auto& loc = buffer->host_location;

    if (loc.alloc_count > 0) {
        return;
    }

    MemoryId other = MemoryId::host();
    if (loc.is_valid && !buffer->find_valid_location(MemoryId::host(), other)) {
        return;
    }

    m_impl->host_state.deallocate(stream_hint, buffer.get());
}

void MemoryManager::invalidate_buffer(const MemoryBuffer& buffer) {
    DeviceStream stream_hint = {};
    buffer->invalidate_all();
    m_impl->reclaim_invalidated(stream_hint, buffer.get());
}

void MemoryManager::trim_device(DeviceId id, size_t bytes_remaining) {
    auto& device = m_impl->device(id);
    DeviceStream stream_hint = {};

    // Evict LRU-eligible locations until `bytes_allocated` is below `bytes_remaining`.
    while (device.bytes_allocated > bytes_remaining) {
        if (!device.try_evict_one(stream_hint)) {
            break;
        }
    }
}

void MemoryManager::make_progress() {
    //
}

KMM_REFCNT_TRAITS_IMPL(MemoryTransactionImpl)
KMM_REFCNT_TRAITS_IMPL(MemoryBufferImpl)
KMM_REFCNT_TRAITS_IMPL(MemoryRequestImpl)

}  // namespace kmm
