#pragma once

#include <future>

#include "kmm/runtime/buffer_registry.hpp"
#include "kmm/runtime/device_resources.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

class Scheduler;
class TaskRecord;

class Task {
    KMM_NOT_COPYABLE_OR_MOVABLE(Task)

  public:
    Task() = default;
    virtual ~Task() = default;
    virtual void start(const DeviceEventSet& input_events) = 0;
    virtual Poll poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) = 0;

    virtual const char* name() const {
        return typeid(*this).name();
    }
};

class JoinTask final: public Task {
  public:
    void start(const DeviceEventSet& input_events) final;
    Poll poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) final;

  private:
    DeviceEventSet m_dependencies;
};

class DeleteBufferTask: public Task {
  public:
    DeleteBufferTask(BufferId buffer_id) : m_buffer_id(buffer_id) {}

    void start(const DeviceEventSet& input_events) final;
    Poll poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) final;

  private:
    BufferId m_buffer_id;
    DeviceEventSet m_dependencies;
};

class HostTask: public Task {
  public:
    HostTask(std::vector<BufferRequirement> buffers) : m_buffers(std::move(buffers)) {}

    void start(const DeviceEventSet& input_events) final;
    Poll poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) final;

  protected:
    virtual std::future<void> submit(
        Scheduler& scheduler,
        std::vector<BufferAccessor> accessors
    ) = 0;
    DeviceEventSet m_dependencies;

  private:
    enum struct Status {
        Init,
        CreateBuffers,
        PollingBuffers,
        PollingDependencies,
        Running,
        Completing,
        Completed
    };

    Status m_status = Status::Init;
    std::future<void> m_future;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
};

class ExecuteHostTask: public HostTask {
  public:
    ExecuteHostTask(  //
        std::unique_ptr<ComputeTask> compute_task,
        std::vector<BufferRequirement> buffers
    ) :
        HostTask(std::move(buffers)),
        m_task(std::move(compute_task)) {}

    std::future<void> submit(Scheduler& scheduler, std::vector<BufferAccessor> accessors) override;

  private:
    std::unique_ptr<ComputeTask> m_task;
};

class CopyHostTask: public HostTask {
  public:
    CopyHostTask(BufferId src_buffer, BufferId dst_buffer, CopyDef definition) :
        HostTask(
            {BufferRequirement {src_buffer, MemoryId::host(), AccessMode::Read},
             BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}}
        ),
        m_copy(definition) {}

    std::future<void> submit(Scheduler& scheduler, std::vector<BufferAccessor> accessors) override;

  private:
    CopyDef m_copy;
};

class ReductionHostTask: public HostTask {
  public:
    ReductionHostTask(BufferId src_buffer, BufferId dst_buffer, ReductionDef definition) :
        HostTask(
            {BufferRequirement {src_buffer, MemoryId::host(), AccessMode::Read},
             BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}}
        ),
        m_reduction(definition) {}

    std::future<void> submit(Scheduler& scheduler, std::vector<BufferAccessor> accessors) override;

  private:
    ReductionDef m_reduction;
};

class FillHostTask: public HostTask {
  public:
    FillHostTask(BufferId dst_buffer, FillDef definition) :
        HostTask({BufferRequirement {dst_buffer, MemoryId::host(), AccessMode::ReadWrite}}),
        m_fill(definition) {}

    std::future<void> submit(Scheduler& scheduler, std::vector<BufferAccessor> accessors) override;

  private:
    FillDef m_fill;
};

class DeviceTask: public Task, public DeviceResourceOperation {
  public:
    DeviceTask(ResourceId resource_id, std::vector<BufferRequirement> buffers) :
        m_resource(resource_id),
        m_buffers(std::move(buffers)) {}

    void start(const DeviceEventSet& input_events) final;
    Poll poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) final;

    ResourceId resource_id() const {
        return m_resource;
    }

  private:
    enum struct Status {
        Init,
        CreateBuffers,
        PollingBuffers,
        PollingDependencies,
        Running,
        Completing,
        Completed
    };

    Status m_status = Status::Init;
    ResourceId m_resource;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEvent m_execution_event;
    DeviceEventSet m_dependencies;
    DeviceEventSet m_local_dependencies;
};

class ExecuteDeviceTask: public DeviceTask {
  public:
    ExecuteDeviceTask(
        ResourceId device_id,
        std::unique_ptr<ComputeTask> compute_task,
        std::vector<BufferRequirement> buffers
    ) :
        DeviceTask(device_id, std::move(buffers)),
        m_task(std::move(compute_task)) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    std::unique_ptr<ComputeTask> m_task;
};

class CopyDeviceTask: public DeviceTask {
  public:
    CopyDeviceTask(
        DeviceId device_id,
        BufferId src_buffer,
        BufferId dst_buffer,
        CopyDef definition
    ) :
        DeviceTask(
            device_id,
            {BufferRequirement {src_buffer, device_id, AccessMode::Read},
             BufferRequirement {dst_buffer, device_id, AccessMode::ReadWrite}}
        ),
        m_copy(definition) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    CopyDef m_copy;
};

class ReductionDeviceTask: public DeviceTask {
  public:
    ReductionDeviceTask(
        DeviceId device_id,
        BufferId src_buffer,
        BufferId dst_buffer,
        ReductionDef definition
    ) :
        DeviceTask(
            device_id,
            {BufferRequirement {src_buffer, device_id, AccessMode::Read},
             BufferRequirement {dst_buffer, device_id, AccessMode::ReadWrite}}
        ),
        m_reduction(std::move(definition)) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    ReductionDef m_reduction;
};

class FillDeviceTask: public DeviceTask {
  public:
    FillDeviceTask(DeviceId device_id, BufferId dst_buffer, FillDef definition) :
        DeviceTask(device_id, {BufferRequirement {dst_buffer, device_id, AccessMode::ReadWrite}}),
        m_fill(std::move(definition)) {}

    void execute(DeviceResource& device, std::vector<BufferAccessor> accessors) final;

  private:
    FillDef m_fill;
};

class PrefetchTask: public Task {
  public:
    PrefetchTask(BufferId buffer_id, MemoryId memory_id) :
        m_buffers {{buffer_id, memory_id, AccessMode::Read}} {}

    void start(const DeviceEventSet& input_events) final;
    Poll poll(TaskRecord& record, Scheduler& scheduler, DeviceEventSet& output_events) final;

  private:
    enum struct Status { Init, Polling, Completing, Completed };

    Status m_status = Status::Init;
    std::vector<BufferRequirement> m_buffers;
    BufferRequestList m_requests;
    DeviceEventSet m_dependencies;
};

std::unique_ptr<Task> build_task_for_command(Command&& command);

}  // namespace kmm
