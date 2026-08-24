#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <utility>

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interface.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/runtime/memory_buffer.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/runtime/reduction_manager.hpp"

namespace kmm {

// A partial buffer handed out by `acquire_partial`: where it lives and the stream it was
// acquired on, so partials can be grouped by location and folded together locally before any
// cross-memory copy.
struct AcquiredPartial {
    MemoryBuffer buffer;
    MemoryId memory_id;
    DeviceStreamId stream_hint;
};

/// Backing state for a single reduction: the destination buffer, the op/type/count describing
/// how partials combine, and the partials handed out by `acquire_partial` that still need
/// folding in. Shared (via `ReductionState`) between the caller and the job that does the
/// folding, so it only holds the data everyone cares about -- none of the polling machinery.
class ReductionStateImpl: public reference_count<ReductionStateImpl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(ReductionStateImpl)

  public:
    ReductionStateImpl(MemoryBuffer home_buffer, ReductionOp op, DataType dtype, size_t count) :
        home_buffer(std::move(home_buffer)),
        op(op),
        dtype(dtype),
        count(count) {}

    MemoryBuffer home_buffer;
    ReductionOp op;
    DataType dtype;
    size_t count;

    std::vector<AcquiredPartial> partials;
    std::unique_ptr<struct ReductionJob> job;
};

KMM_REFCNT_TRAITS_IMPL(ReductionStateImpl)

// Describes folding `count` elements into a destination: `accumulate` is `false` for the first
// source folded into a given destination (overwrites whatever was there) and `true` for every
// subsequent one (combined with what's already there).
static ReductionDescription describe_reduction(
    ReductionOp op,
    DataType dtype,
    size_t count,
    bool accumulate
) {
    auto elem_stride = static_cast<memops_stride_type>(data_type_size(dtype));
    auto group_stride = elem_stride;

    ReductionDescription description;
    description.dtype = dtype;
    description.operation = op;
    description.add_dimension(static_cast<memops_extent_type>(count), group_stride, elem_stride);
    description.reduction_extent = 1;
    description.reduction_stride = elem_stride;
    description.accumulate = accumulate;
    return description;
}

ReductionManager::ReductionManager(
    MemoryManager& memory_manager,
    refcnt_ptr<MemorySystem> memory_system
) :
    m_memory_manager(memory_manager),
    m_memory_system(std::move(memory_system)) {}

ReductionManager::~ReductionManager() = default;

ReductionState ReductionManager::initialize_reduction(
    MemoryBuffer home_buffer,
    ReductionOp op,
    DataType dtype,
    size_t count
) {
    return make_refcnt<ReductionStateImpl>(std::move(home_buffer), op, dtype, count);
}

MemoryBuffer ReductionManager::acquire_partial(
    ReductionState& reduction,
    MemoryId memory_id,
    const DeviceStreamId& stream_hint
) {
    KMM_ASSERT(!is_submitted(reduction));

    // if a stream hint is provided, try to find a existing partial on the same stream and mem.
    if (!stream_hint.is_null()) {
        for (const auto& partials : reduction->partials) {
            if (partials.memory_id == memory_id && partials.stream_hint == stream_hint) {
                return partials.buffer;
            }
        }
    }

    size_t elem_size = data_type_size(reduction->dtype);
    auto layout = BufferLayout {elem_size * reduction->count, elem_size};

    auto data = std::make_unique<FlatDataInterface>(layout, std::move(m_memory_system));
    auto buffer = m_memory_manager.create_buffer(
        std::move(data),
        "reduction-partial:" + reduction->home_buffer->name
    );

    reduction->partials.push_back({buffer, memory_id, stream_hint});
    return buffer;
}

struct PartialFold {
    explicit PartialFold(MemoryId memory_id, MemoryBuffer buffer) :
        memory_id(memory_id),
        buffer(std::move(buffer)) {}

    Poll poll_ready(MemoryManager& manager, const MemoryTransaction& parent) {
        if (!request) {
            auto transaction = manager.create_transaction(parent);
            request = manager.create_request(buffer, memory_id, Access::ReadOnly, transaction);
        }

        return manager.poll_request(DeviceStreamId::null(), request);
    }

    void submit(
        MemoryManager& manager,
        MemorySystem& system,
        void* home_addr,
        const ReductionDescription& description,
        const DeviceEventSet& home_deps
    ) {
        void* local_addr = manager.access_request(request, request_deps).address;
        request_deps.insert(home_deps);

        if (memory_id.is_host()) {
            void* peer_addr = local_addr;
            host_future = std::async(std::launch::async, [peer_addr, home_addr, description] {
                reduce(peer_addr, home_addr, description);
            });
        } else {
            auto event = system.reduce_device(
                memory_id.as_device(),
                reinterpret_cast<GPUDeviceptr>(local_addr),
                reinterpret_cast<GPUDeviceptr>(home_addr),
                description,
                DeviceStreamId::null(),
                request_deps
            );

            completion_deps.insert(event);
        }
    }

    Poll poll_completed(MemoryManager& manager, DeviceEventSet& deps_out) {
        if (host_future.valid()) {
            if (host_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                return Poll::Pending;
            }

            host_future.get();
        }

        deps_out = release(manager);
        return Poll::Ready;
    }

    DeviceEventSet release(MemoryManager& manager, DeviceEventSet release_deps = {}) {
        if (host_future.valid()) {
            host_future.wait();
        }

        release_deps.insert(completion_deps);
        manager.release_request(request, release_deps);
        request = {};
        done = true;
        return release_deps;
    }

    MemoryId memory_id;
    MemoryBuffer buffer;
    MemoryRequest request;
    DeviceEventSet request_deps;
    bool done = false;

  private:
    std::future<void> host_future;
    DeviceEventSet completion_deps;
};

/// The reduction performed for a single memory.
struct ReductionMemoryJob {
    ReductionMemoryJob(
        ReductionState reduction,
        MemoryId memory_id,
        MemoryBuffer home_buffer,
        bool home_buffer_initialized,
        std::vector<MemoryBuffer> buffers = {}
    ) :
        reduction(reduction),
        memory_id(memory_id),
        home_buffer(home_buffer),
        home_buffer_initialized(home_buffer_initialized) {
        partials.reserve(buffers.size());

        for (auto& buffer : buffers) {
            partials.emplace_back(memory_id, std::move(buffer));
        }
    }

    void add_partial(MemoryBuffer buffer) {
        partials.emplace_back(memory_id, std::move(buffer));
    }

    Poll poll(MemoryManager& manager, MemorySystem& system) {
        if (active_index) {
            try_finalize_active(manager);
        }

        if (folded_count == partials.size()) {
            if (home_req) {
                manager.release_request(home_req, chain_deps);
                home_req = {};
            }

            return Poll::Ready;
        }

        if (!home_req) {
            transaction = manager.create_transaction();
            home_req =
                manager.create_request(home_buffer, memory_id, Access::Exclusive, transaction);
        }

        DeviceStream stream_hint = {};
        Poll home_ready = manager.poll_request(DeviceStreamId::null(), home_req);

        // Captured once: after the first partial folds into `home_buffer`, further partials
        // must wait on `chain_deps` (the previous fold's completion) instead, so `home_deps` is
        // deliberately left empty by `try_finalize_active` after that and never repopulated here.
        if (home_ready == Poll::Ready && !home_deps_captured) {
            home_accessor = manager.access_request(home_req, home_deps);
            home_deps_captured = true;
        }

        bool work_remaining = active_index.has_value();

        for (size_t i = 0; i < partials.size(); i++) {
            auto& partial = partials[i];

            if (partial.done || active_index == i) {
                continue;
            }

            Poll peer_ready = partial.poll_ready(manager, transaction);

            if (peer_ready == Poll::Pending || home_ready == Poll::Pending
                || active_index.has_value()) {
                work_remaining = true;
                continue;
            }

            DeviceEventSet wait_deps = chain_deps;
            wait_deps.insert(home_deps);

            if (memory_id.is_host() && !wait_deps.is_empty()) {
                work_remaining = true;
                continue;
            }

            auto description = describe_reduction(
                reduction->op,
                reduction->dtype,
                reduction->count,
                home_buffer_initialized || folded_count > 0
            );

            partial.submit(manager, system, home_accessor.address, description, chain_deps);

            active_index = i;

            if (!try_finalize_active(manager)) {
                work_remaining = true;
            }
        }

        return work_remaining ? Poll::Pending : Poll::Ready;
    }

    void release(MemoryManager& manager) {
        DeviceEventSet deps = chain_deps;

        if (active_index) {
            deps = partials[*active_index].release(manager);
            active_index.reset();
        }

        if (home_req) {
            manager.release_request(home_req, deps);
            home_req = {};
        }

        for (auto& partial : partials) {
            if (!partial.done && partial.request) {
                partial.release(manager);
            }
        }

        reduction = nullptr;
    }

    ReductionState reduction;
    MemoryId memory_id;
    MemoryBuffer home_buffer;
    bool home_buffer_initialized;
    size_t folded_count = 0;
    MemoryRequest home_req;
    MemoryTransaction transaction;
    BufferAccessor home_accessor {};
    bool home_deps_captured = false;
    DeviceEventSet home_deps;
    DeviceEventSet chain_deps;
    std::vector<PartialFold> partials;
    std::optional<size_t> active_index;

  private:
    bool try_finalize_active(MemoryManager& manager) {
        DeviceEventSet release_deps;

        if (partials[*active_index].poll_completed(manager, release_deps) == Poll::Pending) {
            return false;
        }

        home_deps.clear();
        chain_deps = release_deps;
        folded_count++;
        active_index.reset();
        return true;
    }
};

struct ReductionJob {
    ReductionJob(
        ReductionState reduction,
        MemoryId memory_id,
        MemoryBuffer home_buffer,
        std::vector<AcquiredPartial> buffers
    ) {
        std::stable_sort(buffers.begin(), buffers.end(), [](const auto& a, const auto& b) {
            return a.memory_id < b.memory_id;
        });

        size_t index = 0;
        while (index < buffers.size()) {
            auto group_memory_id = buffers[index].memory_id;
            auto group_home_buffer = buffers[index].buffer;
            index++;

            std::vector<MemoryBuffer> partials;

            while (index < buffers.size() && buffers[index].memory_id == group_memory_id) {
                partials.push_back(buffers[index].buffer);
                index++;
            }

            group_jobs.push_back(std::make_unique<ReductionMemoryJob>(
                reduction,
                group_memory_id,
                std::move(group_home_buffer),
                /* home_buffer_initialized = */ true,
                std::move(partials)
            ));
        }

        final_reduction = std::make_unique<ReductionMemoryJob>(
            reduction,
            memory_id,
            std::move(home_buffer),
            /* home_buffer_initialized = */ false
        );
    }

    Poll poll(MemoryManager& manager, MemorySystem& system) {
        Poll result = Poll::Ready;

        for (auto& job : group_jobs) {
            if (!job) {
                // Already finished and handed off to `final_reduction` on an earlier tick.
                continue;
            }

            if (job->poll(manager, system) == Poll::Pending) {
                result = Poll::Pending;
                continue;
            }

            final_reduction->add_partial(std::move(job->home_buffer));
            job = nullptr;
        }

        if (final_reduction->poll(manager, system) == Poll::Pending) {
            result = Poll::Pending;
        }

        return result;
    }

    void release(MemoryManager& manager) {
        for (auto& job : group_jobs) {
            if (job) {
                job->release(manager);
                job = nullptr;
            }
        }

        final_reduction->release(manager);
        final_reduction = nullptr;
    }

  private:
    std::vector<std::unique_ptr<ReductionMemoryJob>> group_jobs;
    std::unique_ptr<ReductionMemoryJob> final_reduction;
};

void ReductionManager::submit_reduction(ReductionState& reduction, MemoryId memory_id) {
    KMM_ASSERT(!is_submitted(reduction));
    reduction->job = std::make_unique<ReductionJob>(
        reduction,
        memory_id,
        reduction->home_buffer,
        reduction->partials
    );
}

bool ReductionManager::is_submitted(const ReductionState& reduction) {
    return bool(reduction) && reduction->job != nullptr;
}

Poll ReductionManager::poll_reduction(ReductionState& reduction) {
    KMM_ASSERT(reduction->job);
    return reduction->job->poll(m_memory_manager, *m_memory_system);
}

void ReductionManager::release_reduction(ReductionState& reduction) {
    // Release any request the job is still holding.
    if (reduction->job) {
        reduction->job->release(m_memory_manager);
        reduction->job = nullptr;
    }

    // Every partial ever handed out by `acquire_partial` is still tracked here, regardless of
    // whether/how the job above grouped and folded it, so release them all directly.
    for (auto& partial : reduction->partials) {
        m_memory_manager.release_buffer(partial.buffer);
    }
}

}  // namespace kmm
