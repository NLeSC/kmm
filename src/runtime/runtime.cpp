#include <thread>

#include "fmt/std.h"
#include "spdlog/spdlog.h"

#include "kmm/runtime/allocators/block.hpp"
#include "kmm/runtime/allocators/caching.hpp"
#include "kmm/runtime/allocators/device.hpp"
#include "kmm/runtime/allocators/system.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

static SystemInfo make_system_info(
    const std::vector<GPUContextHandle>& contexts,
    const RuntimeConfig& config
) {
    spdlog::info("detected {} GPU device(s):", contexts.size());
    std::vector<DeviceInfo> device_infos;

    for (size_t i = 0; i < contexts.size(); i++) {
        auto info = DeviceInfo(DeviceId(i), contexts[i], config.device_concurrent_streams);
        auto memory_gb = static_cast<double>(info.total_memory_size()) / 1e9;

        spdlog::info(" - {} ({:.2} GB)", info.name(), memory_gb);
        device_infos.push_back(info);
    }

    return device_infos;
}

Runtime::Runtime(
    std::vector<GPUContextHandle> contexts,
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::shared_ptr<MemorySystem> memory_system,
    const RuntimeConfig& config
) :
    m_memory_system(memory_system),
    m_memory_manager(std::make_shared<MemoryManager>(memory_system)),
    m_buffer_registry(std::make_shared<BufferRegistry>(m_memory_manager)),
    m_stream_manager(stream_manager),
    m_devices(std::make_shared<DeviceResources>(
        contexts,
        config.device_concurrent_streams,
        m_stream_manager
    )),
    m_info(make_system_info(contexts, config)),
    m_scheduler(m_devices, stream_manager, m_buffer_registry, config.debug_mode) {}

Runtime::~Runtime() {
    shutdown();
}

BufferId Runtime::create_buffer(BufferLayout layout) {
    return this->schedule([&](TaskGraph& g) {  //
        return g.create_buffer(layout);
    });
}

void Runtime::delete_buffer(BufferId id, EventList deps) {
    this->schedule([&](TaskGraph& g) {  //
        g.delete_buffer(id, std::move(deps));
    });
}

void Runtime::check_buffer(BufferId id) {
    std::unique_lock guard {m_mutex};
    m_buffer_registry->get(id);
}

bool Runtime::query_event(EventId event_id, std::chrono::system_clock::time_point deadline) {
    std::unique_lock guard {m_mutex};
    make_progress_impl();

    while (!m_scheduler.is_completed(event_id)) {
        KMM_ASSERT(!m_scheduler.is_idle());
        auto next_update = m_next_updated_planned;

        if (next_update > deadline) {
            return false;
        }

        guard.unlock();
        std::this_thread::sleep_until(next_update);
        guard.lock();

        make_progress_impl();
    }

    return true;
}

bool Runtime::is_idle() {
    std::lock_guard guard {m_mutex};
    return is_idle_impl();
}

void Runtime::trim_memory() {
    std::lock_guard guard {m_mutex};
    m_memory_system->trim_host();
    m_memory_system->trim_device();
}

void Runtime::make_progress() {
    std::lock_guard guard {m_mutex};
    make_progress_impl();
}

void Runtime::shutdown() {
    std::unique_lock guard {m_mutex};
    if (m_has_shutdown) {
        return;
    }

    m_has_shutdown = true;

    while (!is_idle_impl()) {
        make_progress_impl();

        guard.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds {10});
        guard.lock();
    }

    m_stream_manager->wait_until_idle();
}

EventId Runtime::commit_impl(TaskGraph& g) {
    std::vector<TaskGraph::Node> nodes_out;
    std::vector<std::pair<BufferId, BufferLayout>> buffers_out;

    auto barrier_id = m_graph_state.commit(g, nodes_out, buffers_out);

    // Flush all staged buffers to the registry
    for (auto&& [id, layout] : buffers_out) {
        m_buffer_registry->add(id, layout);
    }

    // Flush all events from the DAG builder to the scheduler
    for (auto&& e : nodes_out) {
        m_scheduler.submit(
            e.id,  //
            build_task_for_command(std::move(e.command)),
            std::move(e.dependencies)
        );
    }

    // Plan an update to happen now since we have added new tasks to the scheduler.
    m_next_updated_planned = std::chrono::system_clock::time_point::min();

    return barrier_id;
}

void Runtime::make_progress_impl() {
    static constexpr auto TIMEOUT = std::chrono::microseconds {100};
    auto now = std::chrono::system_clock::now();

    if (m_next_updated_planned > now) {
        return;
    }

    m_next_updated_planned = now + TIMEOUT;
    m_stream_manager->make_progress();
    m_memory_system->make_progress();
    m_scheduler.make_progress();
}

bool Runtime::is_idle_impl() {
    return m_stream_manager->is_idle() && m_scheduler.is_idle()
        && m_memory_manager->is_idle(*m_stream_manager);
}

static size_t compute_device_memory_limit(
    const RuntimeConfig& config,
    const GPUContextHandle& context
) {
    // ignore `device_memory_reserved` if it is zero
    if (config.device_memory_keep_free == 0) {
        return config.device_memory_limit;
    }

    GPUContextGuard guard {context};

    size_t memory_capacity, memory_available;
    KMM_GPU_CHECK(g_mem_get_info(&memory_available, &memory_capacity));

    // Insufficient memory capacity
    if (memory_capacity < config.device_memory_keep_free) {
        spdlog::warn(
            "cannot keep {} bytes available on GPU, memory capacity is only {} bytes",
            config.device_memory_keep_free,
            memory_capacity
        );

        return 0;
    }

    return std::min(
        memory_capacity - config.device_memory_keep_free,  //
        config.device_memory_limit
    );
}

std::unique_ptr<AsyncAllocator> create_device_allocator(
    const RuntimeConfig& config,
    const GPUContextHandle& context,
    std::shared_ptr<DeviceStreamManager> stream_manager
) {
    std::unique_ptr<AsyncAllocator> alloc;
    size_t memory_limit = compute_device_memory_limit(config, context);

    switch (config.device_memory_kind) {
        case DeviceMemoryKind::NoPool:
            return std::make_unique<DeviceMemoryAllocator>(context, stream_manager, memory_limit);
            ;

        case DeviceMemoryKind::CachingPool:
            alloc = std::make_unique<DeviceMemoryAllocator>(context, stream_manager, memory_limit);

            return std::make_unique<CachingAllocator>(std::move(alloc));

        case DeviceMemoryKind::DefaultPool:
            return std::make_unique<DevicePoolAllocator>(
                context,
                stream_manager,
                DevicePoolKind::Default,
                memory_limit
            );

        case DeviceMemoryKind::PrivatePool:
            return std::make_unique<DevicePoolAllocator>(
                context,
                stream_manager,
                DevicePoolKind::Create,
                memory_limit
            );

        default:
            KMM_PANIC("invalid memory kind");
    }
}

std::shared_ptr<Runtime> make_worker(const RuntimeConfig& config) {
    std::unique_ptr<AsyncAllocator> host_mem;
    std::vector<std::unique_ptr<AsyncAllocator>> device_mems;

    auto stream_manager = std::make_shared<DeviceStreamManager>();
    auto contexts = std::vector<GPUContextHandle>();
    auto devices = get_gpu_devices();

    if (devices.empty()) {
        host_mem = std::make_unique<SystemAllocator>(stream_manager, config.host_memory_limit);
    } else if (devices.size() > MAX_DEVICES) {
        throw std::runtime_error(fmt::format("cannot support more than {} GPU(s)", MAX_DEVICES));
    } else {
        for (const auto& device : devices) {
            auto context = GPUContextHandle::retain_primary_context_for_device(device);
            device_mems.push_back(create_device_allocator(config, context, stream_manager));
            contexts.push_back(std::move(context));
        }

        host_mem = std::make_unique<PinnedMemoryAllocator>(
            contexts.at(0),
            stream_manager,
            config.host_memory_limit
        );

        if (config.host_memory_kind == HostMemoryKind::CachingPool) {
            host_mem = std::make_unique<CachingAllocator>(std::move(host_mem));
        }
    }

    if (config.host_memory_block_size > 0) {
        host_mem = std::make_unique<BlockAllocator>(  //
            std::move(host_mem),
            config.host_memory_block_size
        );
    }

    if (config.device_memory_block_size > 0) {
        for (size_t i = 0; i < devices.size(); i++) {
            device_mems[i] = std::make_unique<BlockAllocator>(
                std::move(device_mems[i]),
                config.device_memory_block_size
            );
        }
    }

    auto memory_system = std::make_shared<MemorySystemImpl>(
        stream_manager,
        contexts,
        std::move(host_mem),
        std::move(device_mems)
    );

    return std::make_shared<Runtime>(contexts, stream_manager, memory_system, config);
}
}  // namespace kmm
