#include "kmm/api/runtime_handle.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

struct RuntimeHandle::Impl {
    std::shared_ptr<Runtime> worker;
    SystemInfo info;

    Impl(std::shared_ptr<Runtime> worker, SystemInfo info) : worker(worker), info(info) {}
};

RuntimeHandle::RuntimeHandle(std::shared_ptr<Impl> impl) {
    KMM_ASSERT(impl != nullptr && impl->worker != nullptr);
    m_data = std::move(impl);
}

RuntimeHandle::RuntimeHandle(std::shared_ptr<Runtime> rt) :
    RuntimeHandle(std::make_shared<Impl>(rt, rt->system_info())) {}

RuntimeHandle::RuntimeHandle(Runtime& rt) : RuntimeHandle(rt.shared_from_this()) {}

MemoryId RuntimeHandle::memory_affinity_for_address(const void* address) const {
    if (auto device_opt = get_gpu_device_by_address(address)) {
        const auto& device = worker().system_info().device_by_ordinal(*device_opt);
        return device.memory_id();
    } else {
        return MemoryId::host();
    }
}

EventId RuntimeHandle::join(EventList events) const {
    return worker().schedule([&](TaskGraph& g) { return g.join_events(std::move(events)); });
}

bool RuntimeHandle::is_done(EventId id) const {
    return worker().query_event(id, std::chrono::system_clock::time_point::min());
}

void RuntimeHandle::wait(EventId id) const {
    worker().query_event(id, std::chrono::system_clock::time_point::max());
}

bool RuntimeHandle::wait_until(EventId id, typename std::chrono::system_clock::time_point deadline)
    const {
    return worker().query_event(id, deadline);
}

bool RuntimeHandle::wait_for(EventId id, typename std::chrono::system_clock::duration duration)
    const {
    return worker().query_event(id, std::chrono::system_clock::now() + duration);
}

EventId RuntimeHandle::barrier() const {
    return worker().schedule([&](TaskGraph& g) {  //
        return g.insert_barrier();
    });
}

void RuntimeHandle::synchronize() const {
    wait(barrier());
}

RuntimeHandle RuntimeHandle::constrain_to(std::vector<ResourceId> resources) const {
    return std::make_shared<Impl>(m_data->worker, SystemInfo(m_data->info, std::move(resources)));
}

RuntimeHandle RuntimeHandle::constrain_to(DeviceId device) const {
    return constrain_to(ResourceId(device));
}

RuntimeHandle RuntimeHandle::constrain_to(ResourceId resource) const {
    return constrain_to(std::vector {resource});
}

const SystemInfo& RuntimeHandle::info() const {
    return m_data->info;
}

Runtime& RuntimeHandle::worker() const {
    return *m_data->worker;
}

RuntimeHandle make_runtime(const RuntimeConfig& config) {
    return make_worker(config);
}

}  // namespace kmm
