#include <ankerl/unordered_dense.h>
#include <mutex>
#include <thread>
#include <utility>

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interfaces/flat.hpp"
#include "kmm/runtime/device_data_streams.hpp"
#include "kmm/runtime/memory_buffer.hpp"
#include "kmm/runtime/reduction_manager.hpp"
#include "kmm/runtime/resource.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

struct BufferEntry {
    KMM_NOT_COPYABLE(BufferEntry)

  public:
    MemoryBuffer buffer;
    std::exception_ptr poison = nullptr;
    ReductionState reduction;
    DeviceEventSet last_accesses;

    BufferEntry(MemoryBuffer&& buffer) : buffer(std::move(buffer)) {}

    void record_access(const DeviceEventSet& deps) {
        last_accesses.insert(deps);
    }
};

/// Owns everything a `Runtime` hands out: the machine topology, the stream registry, the
/// physical-memory backend, and the `MemoryManager` built on top of it.
class RuntimeImpl: public reference_count<RuntimeImpl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(RuntimeImpl)

  public:
    RuntimeImpl(const RuntimeConfig& config) :
        system_info {},
        memory_system {std::make_unique<MemorySystem>(system_info, event_registry, config)},
        reduction_manager(memory_manager, memory_system) {}

    // Buffers are normally released one at a time via `Runtime::release_buffer`, but any that
    // are still outstanding when the runtime itself is torn down need to be released here too:
    // otherwise `buffers` would just drop its `MemoryBuffer` refs, and their host/device
    // allocations would never go through `MemoryManager::release_buffer`.
    ~RuntimeImpl() {
        for (auto& [id, entry] : buffers) {
            if (entry.reduction) {
                reduction_manager.release_reduction(entry.reduction);
            }

            memory_manager.release_buffer(std::move(entry.buffer));
        }
    }

    BufferEntry& find_entry(BufferId id) {
        auto it = buffers.find(id);

        if (it == buffers.end()) {
            throw std::runtime_error(fmt::format("could not find buffer {}", id.get()));
        }

        if (it->second.poison) {
            std::rethrow_exception(it->second.poison);
        }

        return it->second;
    }

    MemoryBuffer find_buffer(BufferId id) {
        return find_entry(id).buffer;
    }

    void poison_buffer(BufferId id, std::exception_ptr reason) noexcept {
        auto it = buffers.find(id);

        if (it != buffers.end() && !it->second.poison) {
            it->second.poison = std::move(reason);
        }
    }

    std::chrono::system_clock::time_point poll_once() {
        static constexpr auto poll_interval = std::chrono::milliseconds(5);
        auto now = std::chrono::system_clock::now();

        if (now >= next_poll_deadline) {
            event_registry.make_progress();
            memory_system->make_progress();
            memory_manager.make_progress();

            next_poll_deadline = now + poll_interval;
        }

        return next_poll_deadline;
    }

    template<typename F>
    void poll_until_completion(std::unique_lock<std::mutex>& guard, F callback) {
        if (callback()) {
            return;
        }

        auto deadline = poll_once();

        while (!callback()) {
            guard.unlock();
            std::this_thread::sleep_until(deadline);
            guard.lock();

            deadline = poll_once();
        }
    }

    const SystemInfo system_info;

    std::mutex mutex;
    DeviceEventRegistry event_registry;
    refcnt_ptr<MemorySystem> memory_system;
    MemoryManager memory_manager;
    ReductionManager reduction_manager;
    ankerl::unordered_dense::map<BufferId, BufferEntry> buffers;
    uint64_t next_buffer_id_counter = 0;
    std::chrono::system_clock::time_point next_poll_deadline;
};

KMM_REFCNT_TRAITS_IMPL(RuntimeImpl)

Runtime::Runtime(refcnt_ptr<RuntimeImpl> impl) : m_impl(std::move(impl)) {
    KMM_ASSERT(m_impl);
}

MemorySystem& Runtime::memory_system() noexcept {
    return *m_impl->memory_system;
}

DeviceEventRegistry& Runtime::event_registry() noexcept {
    return m_impl->event_registry;
}

const SystemInfo& Runtime::system_info() const noexcept {
    return m_impl->system_info;
}

std::chrono::system_clock::time_point Runtime::poll_once() {
    std::lock_guard guard {m_impl->mutex};
    return m_impl->poll_once();
}

bool Runtime::poll_until_completion(
    function_ref<bool()> callback,
    const std::chrono::system_clock::time_point deadline
) {
    auto next_poll_at = poll_once();

    while (!callback()) {
        // Next poll is after the deadline. Return false now instead waiting.
        if (next_poll_at > deadline) {
            return false;
        }

        std::this_thread::sleep_until(next_poll_at);
        next_poll_at = poll_once();
    }

    return true;
}

void Runtime::synchronize(const DeviceEvent& e) {
    poll_until_completion([&] { return m_impl->event_registry.is_ready(e); });
}

void Runtime::synchronize(const DeviceEventSet& events) {
    for (const auto& e : events) {
        if (!m_impl->event_registry.is_ready(e)) {
            synchronize(e);
        }
    }
}

void Runtime::synchronize() {
    poll_until_completion([&] { return m_impl->event_registry.is_all_ready(); });
}

BufferId Runtime::create_buffer(
    BufferLayout layout,
    std::string name,
    FillValue fill_value,
    std::optional<MemoryId> home
) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);

    auto system = refcnt_ptr<MemorySystem>(m_impl->memory_system.get(), true);
    auto data =
        std::make_unique<FlatDataInterface>(layout, std::move(system), std::move(fill_value));
    auto id = BufferId(m_impl->next_buffer_id_counter++);

    if (name.empty()) {
        name = std::to_string(id.get());
    }

    auto buffer =
        m_impl->memory_manager.create_buffer(std::move(data), std::move(name), true, home);
    m_impl->buffers.emplace(id, std::move(buffer));
    return id;
}

void Runtime::prefetch_buffer(BufferId id, MemoryId memory_id, bool invalidate_others) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    AccessKind a = invalidate_others ? AccessKind::Exclusive : AccessKind::ReadOnly;
    m_impl->memory_manager.prefetch_buffer(m_impl->find_buffer(id), memory_id, a);
}

void Runtime::poison_buffer(BufferId id, std::exception_ptr reason) noexcept {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    m_impl->poison_buffer(id, std::move(reason));
}

bool Runtime::is_valid(BufferId id, MemoryId memory_id) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    return m_impl->find_buffer(id)->is_valid(memory_id);
}

std::optional<MemoryId> Runtime::find_valid_memory(BufferId id) const {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    const auto& buf = m_impl->find_buffer(id);

    if (buf->is_valid(MemoryId::host())) {
        return MemoryId::host();
    }

    for (size_t i = 0; i < MAX_DEVICES; i++) {
        if (buf->is_valid(MemoryId::device(DeviceId(i)))) {
            return MemoryId::device(DeviceId(i));
        }
    }

    return std::nullopt;
}

std::optional<MemoryId> Runtime::buffer_home(BufferId id) const {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    return m_impl->find_buffer(id)->home_memory_id;
}

bool Runtime::is_allocated(BufferId id, MemoryId memory_id) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    return m_impl->find_buffer(id)->is_allocated(memory_id);
}

void Runtime::try_evict_buffer(BufferId id, MemoryId memory_id) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    m_impl->memory_manager.try_evict_buffer(m_impl->find_buffer(id), memory_id);
}

void Runtime::invalidate_buffer(BufferId id) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    m_impl->memory_manager.invalidate_buffer(m_impl->find_buffer(id));
}

void Runtime::begin_reduction(BufferId id, DataType dtype, ReductionOp op) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    auto& entry = m_impl->find_entry(id);

    if (entry.reduction) {
        throw std::runtime_error(fmt::format("buffer {} is already in reduction mode", id.get()));
    }

    // The number of output elements the partials and the fold operate on is derived from the
    // buffer size: `begin_reduction` fixes the element type, so the buffer holds exactly this
    // many values.
    size_t elem_size = data_type_size(dtype);
    KMM_ASSERT(elem_size > 0 && entry.buffer->size_in_bytes % elem_size == 0);
    size_t count = entry.buffer->size_in_bytes / elem_size;

    entry.reduction =
        m_impl->reduction_manager.initialize_reduction(entry.buffer, op, dtype, count);
}

void Runtime::finalize_reduction(BufferId id, MemoryId memory_id) {
    std::unique_lock<std::mutex> guard(m_impl->mutex);
    auto& entry = m_impl->find_entry(id);
    auto reduction = entry.reduction;

    if (!reduction) {
        throw std::runtime_error(fmt::format("buffer {} is not in reduction mode", id.get()));
    }

    m_impl->reduction_manager.submit_reduction(reduction, memory_id);

    try {
        m_impl->poll_until_completion(guard, [&] {
            return m_impl->reduction_manager.poll_reduction(reduction) == Poll::Ready;
        });
    } catch (...) {
        m_impl->reduction_manager.release_reduction(reduction);
        entry.reduction = nullptr;
        throw;
    }

    m_impl->reduction_manager.release_reduction(reduction);
    entry.reduction = nullptr;
}

void Runtime::rollback_reduction(BufferId id) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    auto& entry = m_impl->find_entry(id);

    if (!entry.reduction) {
        return;
    }

    m_impl->reduction_manager.release_reduction(entry.reduction);
    entry.reduction = nullptr;
}

void Runtime::release_buffer(BufferId id) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);

    auto it = m_impl->buffers.find(id);
    auto& entry = it->second;

    if (entry.reduction) {
        // If the reduction was submitted, we need to release it.
        m_impl->reduction_manager.release_reduction(entry.reduction);
    }

    m_impl->memory_manager.release_buffer(std::move(entry.buffer));
    m_impl->buffers.erase(it);
}

ResourceGrant Runtime::submit(
    ResourceRequest requests,
    std::optional<GPUStreamRef> stream,
    MemoryTransaction parent
) {
    std::unique_lock<std::mutex> guard(m_impl->mutex);

    auto stream_id = DeviceStreamId::null();

    if (stream.has_value()) {
        stream_id = m_impl->event_registry.lookup_or_register_stream(*stream);
    }

    auto transaction = m_impl->memory_manager.create_transaction(parent);

    std::vector<ResourceGrant::Entry> entries;
    entries.reserve(requests.m_requests.size());

    for (const auto& req : requests.m_requests) {
        entries.push_back(ResourceGrant::Entry {req.buffer_id, nullptr, {}});
    }

    for (size_t i = 0; i < entries.size(); i++) {
        auto& it = entries[i];
        const auto& req = requests.m_requests[i];

        try {
            auto& entry = m_impl->find_entry(it.buffer_id);

            if (req.mode == AccessMode::Reduce) {
                if (!entry.reduction) {
                    throw std::runtime_error(fmt::format(
                        "buffer {} is not in reduction mode, call `begin_reduction` before "
                        "reducing into it",
                        it.buffer_id.get()
                    ));
                }

                auto partial = m_impl->reduction_manager.acquire_partial(  //
                    entry.reduction,
                    req.memory_id,
                    stream_id
                );

                it.request = m_impl->memory_manager.create_request(  //
                    partial,
                    req.memory_id,
                    AccessKind::Exclusive,
                    transaction
                );
            } else {
                if (entry.reduction) {
                    throw std::runtime_error(fmt::format(
                        "buffer {} is still in reduction mode, call `finalize_reduction` "
                        "before reading or writing it",
                        it.buffer_id.get()
                    ));
                }

                auto access =
                    req.mode == AccessMode::Read ? AccessKind::ReadOnly : AccessKind::Exclusive;

                it.request = m_impl->memory_manager.create_request(  //
                    entry.buffer,
                    req.memory_id,
                    access,
                    transaction
                );
            }
        } catch (...) {
            for (auto& rollback : entries) {
                if (rollback.request) {
                    m_impl->memory_manager.release_request(std::move(rollback.request));
                }
            }

            throw;
        }
    }

    DeviceEventSet deps;

    try {
        m_impl->poll_until_completion(guard, [&] {
            bool is_ready = true;

            for (auto& entry : entries) {
                if (m_impl->memory_manager.poll_request(stream_id, entry.request)
                    == Poll::Pending) {
                    is_ready = false;
                }
            }

            return is_ready;
        });

        for (auto& entry : entries) {
            entry.accessor = m_impl->memory_manager.access_request(entry.request, deps);
        }

        if (stream_id.is_null()) {
            for (const auto& dep : deps) {
                m_impl->poll_until_completion(guard, [&] {
                    return m_impl->event_registry.is_ready(dep);
                });
            }
        } else {
            m_impl->event_registry.wait_on_event(stream_id, deps);
        }
    } catch (...) {
        for (auto& entry : entries) {
            if (entry.request) {
                m_impl->memory_manager.release_request(std::move(entry.request), {});
            }
        }

        throw;
    }

    return ResourceGrant(std::move(entries), std::move(deps), std::move(transaction));
}

void Runtime::release(ResourceGrant& grant, DeviceEventSet deps) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);

    deps.insert(grant.m_deps);

    for (auto& entry : grant.m_entries) {
        m_impl->find_entry(entry.buffer_id).record_access(deps);
        m_impl->memory_manager.release_request(std::move(entry.request), deps);
    }

    grant.m_entries.clear();
    grant.m_deps.clear();
    grant.m_transaction = {};
}

void Runtime::poison(const ResourceGrant& grant, std::exception_ptr reason) noexcept {
    for (const auto& entry : grant.m_entries) {
        if (entry.accessor.is_writable) {
            poison_buffer(entry.buffer_id, reason);
        }
    }
}

static DeviceEvent do_copy(
    std::unique_lock<std::mutex>& guard,
    RuntimeImpl* impl,
    BufferAccessor dst_access,
    BufferAccessor src_access,
    const CopyDescription& description,
    const DeviceEventSet& deps,
    const DeviceStreamId& stream_hint = {}
) {
    KMM_ASSERT(range(dst_access.size_in_bytes).contains(description.dst_range()));
    KMM_ASSERT(range(src_access.size_in_bytes).contains(description.src_range()));

    auto dst_memory_id = dst_access.memory_id;
    auto src_memory_id = src_access.memory_id;

    DeviceEvent event;

    if (dst_memory_id.is_device() && src_memory_id.is_device()) {
        event = impl->memory_system->copy_device_to_device(
            src_memory_id.as_device(),
            dst_memory_id.as_device(),
            g_device_ptr_t(src_access.address) + description.src_offset,
            g_device_ptr_t(dst_access.address) + description.dst_offset,
            description.element_size,
            stream_hint,
            deps
        );
    } else if (dst_memory_id.is_device() && src_memory_id.is_host()) {
        event = impl->memory_system->copy_host_to_device(
            dst_memory_id.as_device(),
            reinterpret_cast<const std::byte*>(src_access.address) + description.src_offset,
            g_device_ptr_t(dst_access.address) + description.dst_offset,
            description.element_size,
            stream_hint,
            deps
        );
    } else if (dst_memory_id.is_host() && src_memory_id.is_device()) {
        event = impl->memory_system->copy_device_to_host(
            src_memory_id.as_device(),
            g_device_ptr_t(src_access.address) + description.src_offset,
            reinterpret_cast<std::byte*>(dst_access.address) + description.dst_offset,
            description.element_size,
            stream_hint,
            deps
        );
    } else {
        // TOOD: maybe use a thread pool?
        auto fut = std::async([=] { copy(src_access.address, dst_access.address, description); });

        impl->poll_until_completion(guard, [&] {
            if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::timeout) {
                return false;
            }

            fut.get();
            return true;
        });
    }

    return event;
}

DeviceEvent Runtime::submit_copy(
    BufferId dst_id,
    BufferId src_id,
    CopyDescription description,
    MemoryId memory_id,
    std::optional<GPUStreamRef> stream,
    MemoryTransaction parent
) {
    std::unique_lock<std::mutex> guard(m_impl->mutex);
    description = description.simplify();

    auto stream_hint = DeviceStreamId::null();

    if (stream.has_value()) {
        stream_hint = m_impl->event_registry.lookup_or_register_stream(*stream);
    }

    if (description.num_dims > 0) {
        throw std::runtime_error("only copy of dimensionality N=1 are supported for now");
    }

    auto& dst_entry = m_impl->find_entry(dst_id);
    auto& src_entry = m_impl->find_entry(src_id);

    bool same_buffer = dst_id == src_id;

    MemoryId dst_memory_id = memory_id;
    MemoryId src_memory_id =
        same_buffer ? dst_memory_id : src_entry.buffer->find_preferred_location(memory_id);

    auto transaction = m_impl->memory_manager.create_transaction(parent);

    MemoryRequest dst_req = m_impl->memory_manager.create_request(  //
        dst_entry.buffer,
        dst_memory_id,
        AccessKind::Exclusive,
        transaction
    );
    MemoryRequest src_req = dst_req;

    if (!same_buffer) {
        try {
            src_req = m_impl->memory_manager.create_request(  //
                src_entry.buffer,
                src_memory_id,
                AccessKind::ReadOnly,
                transaction
            );
        } catch (...) {
            // we must release the other request as well.
            m_impl->memory_manager.release_request(std::move(dst_req));
            throw;
        }
    }

    DeviceEventSet deps;
    DeviceEvent event;

    try {
        m_impl->poll_until_completion(guard, [&] {
            auto dst_status = m_impl->memory_manager.poll_request(stream_hint, dst_req);
            auto src_status = m_impl->memory_manager.poll_request(stream_hint, src_req);
            return src_status == Poll::Ready && dst_status == Poll::Ready;
        });

        auto dst_accessor = m_impl->memory_manager.access_request(dst_req, deps);
        auto src_accessor = m_impl->memory_manager.access_request(src_req, deps);

        event = do_copy(
            guard,
            m_impl.get(),
            dst_accessor,
            src_accessor,
            description,
            deps,
            stream_hint
        );
    } catch (...) {
        if (!same_buffer) {
            m_impl->memory_manager.release_request(src_req);
            m_impl->memory_manager.release_request(dst_req);
        } else {
            m_impl->memory_manager.release_request(dst_req);
        }

        throw;
    }

    if (!same_buffer) {
        m_impl->memory_manager.release_request(src_req, event);
        m_impl->memory_manager.release_request(dst_req, event);
    } else {
        m_impl->memory_manager.release_request(dst_req, event);
    }

    return event;
}

static DeviceEvent do_reduce(
    std::unique_lock<std::mutex>& guard,
    RuntimeImpl* impl,
    BufferAccessor dst_access,
    BufferAccessor src_access,
    const ReductionDescription& description,
    const DeviceEventSet& deps,
    const DeviceStreamId& stream_hint = {}
) {
    KMM_ASSERT(range(dst_access.size_in_bytes).contains(description.dst_range()));
    KMM_ASSERT(range(src_access.size_in_bytes).contains(description.src_range()));

    auto memory_id = dst_access.memory_id;
    DeviceEvent event;

    if (memory_id.is_device()) {
        event = impl->memory_system->reduce_device(
            memory_id.as_device(),
            g_device_ptr_t(src_access.address),
            g_device_ptr_t(dst_access.address),
            description,
            stream_hint,
            deps
        );
    } else {
        // TODO: maybe use a thread pool?
        auto fut = std::async([=] { reduce(src_access.address, dst_access.address, description); });

        impl->poll_until_completion(guard, [&] {
            if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::timeout) {
                return false;
            }

            fut.get();
            return true;
        });
    }

    return event;
}

DeviceEvent Runtime::submit_reduction(
    BufferId dst_id,
    BufferId src_id,
    ReductionDescription description,
    MemoryId memory_id,
    std::optional<GPUStreamRef> stream,
    MemoryTransaction parent
) {
    std::unique_lock<std::mutex> guard(m_impl->mutex);
    description = description.simplify();

    auto stream_hint = DeviceStreamId::null();

    if (stream.has_value()) {
        stream_hint = m_impl->event_registry.lookup_or_register_stream(*stream);
    }

    auto& dst_entry = m_impl->find_entry(dst_id);
    auto& src_entry = m_impl->find_entry(src_id);

    if (src_entry.reduction) {
        throw std::runtime_error(fmt::format(
            "buffer {} is still in reduction mode and cannot be a reduction source, call "
            "`finalize_reduction` before reading it",
            src_id.get()
        ));
    }

    // If the destination is mid-reduction its home buffer must not be written directly (the
    // outstanding partials have not been folded in yet). Route this contribution into a fresh
    // partial of the ongoing reduction instead; `finalize_reduction` folds it in with the rest.
    MemoryBuffer dst_buffer = dst_entry.buffer;

    if (dst_entry.reduction) {
        if (m_impl->reduction_manager.is_submitted(dst_entry.reduction)) {
            throw std::runtime_error(fmt::format(
                "buffer {} reduction has already been finalized, cannot reduce into it",
                dst_id.get()
            ));
        }

        m_impl->reduction_manager
            .check_compatible(dst_entry.reduction, description.operation, description.dtype);

        dst_buffer =
            m_impl->reduction_manager.acquire_partial(dst_entry.reduction, memory_id, stream_hint);

        // The partial is freshly allocated: there is nothing in it to combine with.
        description.accumulate = false;
    }

    auto transaction = m_impl->memory_manager.create_transaction(parent);

    MemoryRequest dst_req = m_impl->memory_manager.create_request(  //
        dst_buffer,
        memory_id,
        AccessKind::Exclusive,
        transaction
    );
    MemoryRequest src_req;

    try {
        src_req = m_impl->memory_manager.create_request(  //
            src_entry.buffer,
            memory_id,
            AccessKind::ReadOnly,
            transaction
        );
    } catch (...) {
        m_impl->memory_manager.release_request(std::move(dst_req));
        throw;
    }

    DeviceEventSet deps;
    DeviceEvent event;

    try {
        m_impl->poll_until_completion(guard, [&] {
            auto dst_status = m_impl->memory_manager.poll_request(stream_hint, dst_req);
            auto src_status = m_impl->memory_manager.poll_request(stream_hint, src_req);
            return src_status == Poll::Ready && dst_status == Poll::Ready;
        });

        auto dst_accessor = m_impl->memory_manager.access_request(dst_req, deps);
        auto src_accessor = m_impl->memory_manager.access_request(src_req, deps);

        event = do_reduce(
            guard,
            m_impl.get(),
            dst_accessor,
            src_accessor,
            description,
            deps,
            stream_hint
        );
    } catch (...) {
        m_impl->memory_manager.release_request(src_req);
        m_impl->memory_manager.release_request(dst_req);
        throw;
    }

    m_impl->memory_manager.release_request(src_req, event);
    m_impl->memory_manager.release_request(dst_req, event);

    return event;
}

Runtime make_runtime(const RuntimeConfig& config) {
    return Runtime(std::make_unique<RuntimeImpl>(config));
}

}  // namespace kmm
