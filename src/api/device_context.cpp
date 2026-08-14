#include "kmm/api/device_context.hpp"

namespace kmm {

DeviceContext::DeviceContext(Runtime runtime, DeviceId device_id, DeviceEventSet deps) :
    Context(runtime),
    m_device_id(device_id),
    m_stream(std::make_shared<CUDAStream>(runtime.system_info().device(device_id).context())) {}

DeviceStream DeviceContext::stream() const noexcept {
    auto registry = Runtime(runtime()).event_registry();
    return {registry, registry.lookup_or_register_stream(*m_stream)};
}

std::optional<CUDAStreamRef> DeviceContext::submit_stream() noexcept {
    return *m_stream;
}

DeviceContext Context::gpu(DeviceId device_id) {
    return DeviceContext(m_runtime, device_id);
}

}  // namespace kmm
