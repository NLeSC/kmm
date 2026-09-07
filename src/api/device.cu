#include "kmm/api/device.hpp"

namespace kmm {

Device::Device(Runtime runtime, DeviceId device_id, MemoryTransaction transaction) :
    Context(runtime, std::move(transaction)),
    m_device_id(device_id),
    m_stream(nullptr) {
    auto* context = runtime.system_info().device(device_id).context();
    m_stream = std::make_shared<GPUStreamOwner>(context);
}

DeviceStream Device::stream() const noexcept {
    auto registry = Runtime(runtime()).event_registry();
    return {registry, registry.lookup_or_register_stream(*m_stream)};
}

Device Context::gpu(DeviceId device_id) {
    return Device(m_runtime, device_id, m_transaction);
}

}  // namespace kmm
