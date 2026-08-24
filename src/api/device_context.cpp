#include "kmm/api/device_context.hpp"

namespace kmm {

DeviceContext::DeviceContext(Runtime runtime, DeviceId device_id, MemoryTransaction transaction) :
    Context(runtime, std::move(transaction)),
    m_device_id(device_id),
    m_stream(std::make_shared<GPUStreamOwner>(runtime.system_info().device(device_id).context())) {}

DeviceStream DeviceContext::stream() const noexcept {
    auto registry = Runtime(runtime()).event_registry();
    return {registry, registry.lookup_or_register_stream(*m_stream)};
}

DeviceContext Context::gpu(DeviceId device_id) {
    return DeviceContext(m_runtime, device_id, m_transaction);
}

}  // namespace kmm
