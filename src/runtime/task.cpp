#include "kmm/memops/gpu_copy.hpp"
#include "kmm/memops/gpu_fill.hpp"
#include "kmm/memops/gpu_reduction.hpp"
#include "kmm/memops/host_copy.hpp"
#include "kmm/memops/host_fill.hpp"
#include "kmm/memops/host_reduction.hpp"
#include "kmm/runtime/scheduler.hpp"
#include "kmm/runtime/task.hpp"

namespace kmm {

static PoisonException make_poison_exception(TaskRecord& record, const std::exception& error) {
    if (const auto* reason = dynamic_cast<const PoisonException*>(&error)) {
        return *reason;
    }

    return fmt::format("task {} failed due to error: {}", record.id(), error.what());
}

void JoinTask::start(const DeviceEventSet& input_events) {
    m_dependencies = input_events;
}

Poll JoinTask::poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) {
    output_events.insert(std::move(m_dependencies));
    return Poll::Ready;
}

void DeleteBufferTask::start(const DeviceEventSet& input_events) {
    m_dependencies = input_events;
}

Poll DeleteBufferTask::poll(
    TaskRecord& record,
    Scheduler& scheduler,
    DeviceEventSet& output_events
) {
    scheduler.buffers().remove(m_buffer_id);
    output_events.insert(m_dependencies);
    return Poll::Ready;
}

void HostTask::start(const DeviceEventSet& input_events) {
    KMM_ASSERT(m_status == Status::Init);
    m_dependencies = input_events;
    m_status = Status::CreateBuffers;
}

Poll HostTask::poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) {
    if (m_status == Status::CreateBuffers) {
        try {
            m_requests = scheduler.buffers().create_requests(m_buffers);
            m_status = Status::PollingBuffers;
        } catch (const std::exception& e) {
            scheduler.buffers().poison_all(m_buffers, make_poison_exception(record, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingBuffers) {
        try {
            if (scheduler.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
                return Poll::Pending;
            }

            m_status = Status::PollingDependencies;
        } catch (const std::exception& e) {
            scheduler.buffers().poison_all(m_buffers, make_poison_exception(record, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingDependencies) {
        try {
            if (!scheduler.streams().is_ready(m_dependencies)) {
                return Poll::Pending;
            }

            m_future = submit(scheduler, scheduler.buffers().access_requests(m_requests));
            m_status = Status::Running;
        } catch (const std::exception& e) {
            scheduler.buffers().poison_all(m_buffers, make_poison_exception(record, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::Running) {
        try {
            if (m_future.wait_for(std::chrono::seconds(0)) == std::future_status::timeout) {
                return Poll::Pending;
            }
        } catch (const std::exception& e) {
            scheduler.buffers().poison_all(m_buffers, make_poison_exception(record, e));
        }

        m_status = Status::Completing;
    }

    if (m_status == Status::Completing) {
        scheduler.buffers().release_requests(m_requests);
        m_status = Status::Completed;
    }

    return Poll::Ready;
}

std::future<void> ExecuteHostTask::submit(
    Scheduler& scheduler,
    std::vector<BufferAccessor> accessors
) {
    KMM_ASSERT(m_task != nullptr);
    auto* task = m_task.get();

    return std::async(std::launch::async, [=] {
        auto host = HostResource {};
        auto context = TaskContext {std::move(accessors)};
        task->execute(host, context);
    });
}

std::future<void> CopyHostTask::submit(
    Scheduler& scheduler,
    std::vector<BufferAccessor> accessors
) {
    KMM_ASSERT(accessors[0].layout.size_in_bytes >= m_copy.minimum_source_bytes_needed());
    KMM_ASSERT(accessors[1].layout.size_in_bytes >= m_copy.minimum_destination_bytes_needed());
    KMM_ASSERT(accessors[1].is_writable);

    return std::async(std::launch::async, [=] {
        execute_copy(accessors[0].address, accessors[1].address, m_copy);
    });
}

std::future<void> ReductionHostTask::submit(
    Scheduler& scheduler,
    std::vector<BufferAccessor> accessors
) {
    return std::async(std::launch::async, [=] {
        execute_reduction(accessors[0].address, accessors[1].address, m_reduction);
    });
}

std::future<void> FillHostTask::submit(
    Scheduler& scheduler,
    std::vector<BufferAccessor> accessors
) {
    return std::async(std::launch::async, [=] { execute_fill(accessors[0].address, m_fill); });
}

void DeviceTask::start(const DeviceEventSet& input_events) {
    KMM_ASSERT(m_status == Status::Init);
    m_dependencies = input_events;
    m_status = Status::CreateBuffers;
}

Poll DeviceTask::poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) {
    if (m_status == Status::CreateBuffers) {
        try {
            m_requests = scheduler.buffers().create_requests(m_buffers);
            m_status = Status::PollingBuffers;
        } catch (const std::exception& e) {
            scheduler.buffers().poison_all(m_buffers, make_poison_exception(record, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingBuffers) {
        try {
            if (scheduler.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
                return Poll::Pending;
            }

            // Remove the `local` events from the list of dependencies. These events
            // have the same context as the current device, and thus can be directly put
            // as dependencies on the current device stream.
            m_local_dependencies = m_dependencies.extract_events_for_context(
                scheduler.streams(),
                scheduler.devices().context(m_resource.as_device())
            );

            output_events.insert(m_local_dependencies);
            m_status = Status::PollingDependencies;
        } catch (const std::exception& e) {
            scheduler.buffers().poison_all(m_buffers, make_poison_exception(record, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::PollingDependencies) {
        if (!scheduler.streams().is_ready(m_dependencies)) {
            return Poll::Pending;
        }

        m_status = Status::Running;
    }

    if (m_status == Status::Running) {
        try {
            m_execution_event = scheduler.devices().submit(
                m_resource.as_device(),
                m_resource.stream_affinity(),
                m_local_dependencies,
                *this,
                scheduler.buffers().access_requests(m_requests)
            );

            output_events.insert(m_execution_event);
            m_status = Status::Completing;
        } catch (const std::exception& e) {
            scheduler.buffers().poison_all(m_buffers, make_poison_exception(record, e));
            m_status = Status::Completing;
        }
    }

    if (m_status == Status::Completing) {
        scheduler.buffers().release_requests(m_requests, m_execution_event);
        m_status = Status::Completed;
    }

    return Poll::Ready;
}

void ExecuteDeviceTask::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
    KMM_ASSERT(m_task != nullptr);
    auto context = TaskContext {std::move(accessors)};
    m_task->execute(device, context);
}

void CopyDeviceTask::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
    KMM_ASSERT(accessors[0].layout.size_in_bytes >= m_copy.minimum_source_bytes_needed());
    KMM_ASSERT(accessors[1].layout.size_in_bytes >= m_copy.minimum_destination_bytes_needed());
    KMM_ASSERT(accessors[1].is_writable);

    execute_gpu_d2d_copy_async(
        device,
        reinterpret_cast<g_device_ptr_t>(accessors[0].address),
        reinterpret_cast<g_device_ptr_t>(accessors[1].address),
        m_copy
    );
}

void ReductionDeviceTask::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
    execute_gpu_reduction_async(
        device,
        reinterpret_cast<g_device_ptr_t>(accessors[0].address),
        reinterpret_cast<g_device_ptr_t>(accessors[1].address),
        m_reduction
    );
}

void FillDeviceTask::execute(DeviceResource& device, std::vector<BufferAccessor> accessors) {
    execute_gpu_fill_async(device, reinterpret_cast<g_device_ptr_t>(accessors[0].address), m_fill);
}

void PrefetchTask::start(const DeviceEventSet& input_events) {
    m_dependencies = input_events;
}

Poll PrefetchTask::poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) {
    if (m_status == Status::Init) {
        m_requests = scheduler.buffers().create_requests(m_buffers);
        m_status = Status::Polling;
    }

    if (m_status == Status::Polling) {
        if (scheduler.buffers().poll_requests(m_requests, m_dependencies) == Poll::Pending) {
            return Poll::Pending;
        }

        scheduler.buffers().release_requests(m_requests);
        m_status = Status::Completing;
    }

    if (m_status == Status::Completing) {
        if (!scheduler.streams().is_ready(m_dependencies)) {
            return Poll::Pending;
        }

        m_status = Status::Completed;
    }

    return Poll::Ready;
}

std::unique_ptr<Task> build_task_for_command(Command&& command) {
    if (std::get_if<CommandEmpty>(&command) != nullptr) {
        return std::make_unique<JoinTask>();

    } else if (const auto* e = std::get_if<CommandBufferDelete>(&command)) {
        return std::make_unique<DeleteBufferTask>(e->id);

    } else if (const auto* e = std::get_if<CommandPrefetch>(&command)) {
        return std::make_unique<PrefetchTask>(e->buffer_id, e->memory_id);

    } else if (auto* e = std::get_if<CommandExecute>(&command)) {
        auto proc = e->processor_id;

        if (proc.is_device()) {
            return std::make_unique<ExecuteDeviceTask>(proc, std::move(e->task), e->buffers);
        } else {
            return std::make_unique<ExecuteHostTask>(std::move(e->task), e->buffers);
        }

    } else if (const auto* e = std::get_if<CommandCopy>(&command)) {
        auto src_mem = e->src_memory;
        auto dst_mem = e->dst_memory;

        if (src_mem.is_host() && dst_mem.is_host()) {
            return std::make_unique<CopyHostTask>(e->src_buffer, e->dst_buffer, e->definition);
        } else if (dst_mem.is_device()) {
            return std::make_unique<CopyDeviceTask>(
                dst_mem.as_device(),
                e->src_buffer,
                e->dst_buffer,
                e->definition
            );
        } else if (src_mem.is_device()) {
            return std::make_unique<CopyDeviceTask>(
                src_mem.as_device(),
                e->src_buffer,
                e->dst_buffer,
                e->definition
            );
        } else {
            KMM_PANIC("unsupported copy");
        }

    } else if (const auto* e = std::get_if<CommandReduction>(&command)) {
        auto memory_id = e->memory_id;

        if (memory_id.is_device()) {
            return std::make_unique<ReductionDeviceTask>(
                memory_id.as_device(),
                e->src_buffer,
                e->dst_buffer,
                std::move(e->definition)
            );
        } else {
            return std::make_unique<ReductionHostTask>(
                e->src_buffer,
                e->dst_buffer,
                std::move(e->definition)
            );
        }

    } else if (const auto* e = std::get_if<CommandFill>(&command)) {
        auto memory_id = e->memory_id;

        if (memory_id.is_device()) {
            return std::make_unique<FillDeviceTask>(
                memory_id.as_device(),
                e->dst_buffer,
                std::move(e->definition)
            );
        } else {
            return std::make_unique<FillHostTask>(e->dst_buffer, std::move(e->definition));
        }

    } else {
        KMM_PANIC_FMT("could not handle unknown command: {}", command);
    }
}

}  // namespace kmm
