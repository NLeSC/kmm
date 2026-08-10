#include <ankerl/unordered_dense.h>
#include <mutex>
#include <thread>
#include <utility>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/data_interface.hpp"
#include "kmm/runtime/memory_buffer.hpp"
#include "kmm/runtime/reduction_manager.hpp"
#include "kmm/runtime/requisition.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

struct BufferEntry {
    MemoryBuffer buffer;
    std::exception_ptr poison = nullptr;
    ReductionState reduction;
    DeviceEventSet last_accesses;

    BufferEntry(MemoryBuffer&& buffer) : buffer(std::move(buffer)) {}

    void record_access(MemoryId memory_id, AccessMode mode, const DeviceEventSet& deps) {
        last_accesses.insert(deps);
    }
};

/// Owns everything a `Runtime` hands out: the machine topology, the stream registry, the
/// physical-memory backend, and the `MemoryManager` built on top of it.
class RuntimeImpl: public reference_count<RuntimeImpl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(RuntimeImpl)

  public:
    RuntimeImpl(size_t host_memory_limit, size_t device_memory_limit) :
        memory_system(make_refcnt<MemorySystem>(
            system_info,
            &stream_registry,
            host_memory_limit,
            device_memory_limit
        )),
        reduction_manager(memory_manager, memory_system) {}

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
            stream_registry.make_progress();
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

    SystemInfo system_info;

    std::mutex mutex;
    DeviceStreamRegistry stream_registry;
    refcnt_ptr<MemorySystem> memory_system;
    MemoryManager memory_manager;
    ReductionManager reduction_manager;
    ankerl::unordered_dense::map<BufferId, BufferEntry> buffers;
    uint64_t next_buffer_id_counter = 0;
    std::chrono::system_clock::time_point next_poll_deadline;
};

Runtime::Runtime(size_t host_memory_limit, size_t device_memory_limit) :
    m_impl(make_refcnt<RuntimeImpl>(host_memory_limit, device_memory_limit)) {}

MemorySystem& Runtime::memory_system() noexcept {
    return *m_impl->memory_system;
}

DeviceStreamRegistry& Runtime::stream_registry() noexcept {
    return m_impl->stream_registry;
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
    poll_until_completion([&] { return e.is_ready(); });
}

void Runtime::synchronize(const DeviceEventSet& events) {
    for (const auto& e : events) {
        if (!e.is_ready()) {
            synchronize(e);
        }
    }
}

BufferId Runtime::create_buffer(BufferLayout layout, std::string name, FillValue fill_value) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);

    auto system = refcnt_ptr<MemorySystem>(m_impl->memory_system.get(), true);
    auto data =
        std::make_unique<FlatDataInterface>(layout, std::move(system), std::move(fill_value));
    auto buffer = m_impl->memory_manager.create_buffer(std::move(data), std::move(name));

    auto id = BufferId(m_impl->next_buffer_id_counter++);
    m_impl->buffers.emplace(id, std::move(buffer));
    return id;
}

void Runtime::prefetch_buffer(BufferId id, MemoryId memory_id, bool invalid_others) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    Access a = invalid_others ? Access::Exclusive : Access::ReadOnly;
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

    entry.reduction = m_impl->reduction_manager.initialize_reduction(entry.buffer, op, dtype, 0);
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
    auto entry = std::move(it->second);
    m_impl->buffers.erase(it);

    if (entry.reduction) {
        // If the reduction was submitted, we need to release it.
        m_impl->reduction_manager.release_reduction(entry.reduction);
    }

    m_impl->memory_manager.release_buffer(std::move(entry.buffer));
}

void Runtime::submit(Requisition& req) {
    std::unique_lock<std::mutex> guard(m_impl->mutex);
    KMM_ASSERT(req.m_state == RequisitionState::Building);

    auto transaction = m_impl->memory_manager.create_transaction(req.m_parent);

    DeviceStream stream_hint = {};
    KMM_ASSERT(!stream_hint.is_null());

    for (size_t i = 0; i < req.m_entries.size(); i++) {
        auto& it = req.m_entries[i];

        try {
            auto& entry = m_impl->find_entry(it.buffer_id);

            if (it.mode == AccessMode::Reduce) {
                if (!entry.reduction) {
                    throw std::runtime_error(fmt::format(
                        "buffer {} is not in reduction mode, call `begin_reduction` before "
                        "reducing into it",
                        it.buffer_id.get()
                    ));
                }

                auto partial = m_impl->reduction_manager.acquire_partial(  //
                    entry.reduction,
                    it.memory_id,
                    stream_hint
                );

                it.request = m_impl->memory_manager.create_request(  //
                    partial,
                    it.memory_id,
                    Access::Exclusive,
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

                auto access = it.mode == AccessMode::Read ? Access::ReadOnly : Access::Exclusive;

                it.request = m_impl->memory_manager.create_request(  //
                    entry.buffer,
                    it.memory_id,
                    access,
                    transaction
                );
            }
        } catch (...) {
            for (auto& rollback : req.m_entries) {
                if (rollback.request) {
                    m_impl->memory_manager.release_request(std::move(rollback.request));
                }
            }

            throw;
        }
    }

    req.m_state = RequisitionState::Submitted;

    m_impl->poll_until_completion(guard, [&] {
        bool is_ready = true;

        for (auto& entry : req.m_entries) {
            if (m_impl->memory_manager.poll_request(stream_hint, entry.request, req.m_deps)
                == Poll::Pending) {
                is_ready = false;
            }
        }

        return is_ready;
    });

    req.m_state = RequisitionState::Ready;
}

BufferAccessor Runtime::accessor(const kmm::RequisitionDep& dep) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    return m_impl->memory_manager.access_request(dep.request);
}

void Runtime::release(Requisition& req, DeviceEventSet deps) {
    std::lock_guard<std::mutex> guard(m_impl->mutex);
    KMM_ASSERT(req.m_state != RequisitionState::Released);
    req.m_state = RequisitionState::Released;

    if (!req.m_stream.is_null()) {
        req.m_stream.wait_on_default_stream();
        deps.insert(req.m_stream.record_event());
    }

    deps.insert(req.dependencies());

    for (auto& entry : req.m_entries) {
        m_impl->find_entry(entry.buffer_id).record_access(entry.memory_id, entry.mode, deps);
        m_impl->memory_manager.release_request(std::move(entry.request), deps);
    }
}

KMM_REFCNT_TRAITS_IMPL(RuntimeImpl)

}  // namespace kmm
