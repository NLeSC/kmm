#include <algorithm>

#include "spdlog/spdlog.h"

#include "kmm/runtime/device_resources.hpp"

namespace kmm {

struct DeviceResources::Device {
    KMM_NOT_COPYABLE_OR_MOVABLE(Device)

  public:
    Device(GPUContextHandle context) : context(context) {}

    GPUContextHandle context;
    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<size_t> last_used_streams;
};

struct DeviceResources::Stream {
    KMM_NOT_COPYABLE_OR_MOVABLE(Stream)

  public:
    Stream(
        DeviceId device_id,
        DeviceStream stream,
        GPUContextHandle context,
        g_stream_t gpu_stream
    ) :
        context(context),
        resource(DeviceInfo(device_id, context), context, gpu_stream),
        stream(stream) {}

    GPUContextHandle context;
    DeviceResource resource;
    DeviceStream stream;
    DeviceEvent last_event;
};

DeviceResources::DeviceResources(
    std::vector<GPUContextHandle> contexts,
    size_t streams_per_context,
    std::shared_ptr<DeviceStreamManager> stream_manager
) :
    m_stream_manager(stream_manager),
    m_streams_per_device(streams_per_context) {
    KMM_ASSERT(m_streams_per_device > 0);

    for (size_t i = 0; i < contexts.size(); i++) {
        m_devices.emplace_back(std::make_unique<Device>(contexts[i]));

        for (size_t j = 0; j < m_streams_per_device; j++) {
            auto stream = stream_manager->create_stream(contexts[i]);
            auto s = std::make_unique<Stream>(
                DeviceId(i),
                stream,
                contexts[i],
                stream_manager->get(stream)
            );

            m_devices[i]->streams.emplace_back(std::move(s));
            m_devices[i]->last_used_streams.push_back(j);
        }
    }
}

DeviceResources::~DeviceResources() {
    for (const auto& device : m_devices) {
        for (const auto& e : device->streams) {
            m_stream_manager->wait_until_ready(e->stream);
        }
    }
}

size_t DeviceResources::num_contexts() const {
    return m_devices.size();
}

GPUContextHandle DeviceResources::context(DeviceId device_id) {
    KMM_ASSERT(device_id < m_devices.size());
    return m_devices[device_id]->context;
}

DeviceResources::Stream* DeviceResources::select_stream_for_operation(
    DeviceId device_id,
    DeviceStreamSet stream_hint,
    const DeviceEventSet& deps
) {
    static constexpr size_t INVALID = ~size_t(0);
    KMM_ASSERT(device_id < m_devices.size());
    auto& device = *m_devices[device_id];
    size_t stream_index = INVALID;

    // Limit available streams to the range 0...streams.size()
    stream_hint &= DeviceStreamSet::range(0, device.streams.size());

    // No stream given, set it to all streams
    if (stream_hint.is_empty()) {
        stream_hint = DeviceStreamSet::all();
    }

    // Case 1: Find a stream that contains one of the dependencies
    if (stream_index == INVALID) {
        for (auto i : device.last_used_streams) {
            auto e = device.streams[i]->last_event;

            if (stream_hint.contains(i) && std::find(deps.begin(), deps.end(), e) != deps.end()) {
                stream_index = i;
                break;
            }
        }
    }

    // Case 2: Otherwise, find the last used stream that is contained in `stream_hint`
    if (stream_index == INVALID) {
        for (auto i : device.last_used_streams) {
            if (stream_hint.contains(i)) {
                stream_index = i;
                break;
            }
        }
    }

    // Case 3: Otherwise, select the last used stream
    if (stream_index == INVALID) {
        stream_index = device.last_used_streams[0];
    }

    // Push this stream to the back
    auto it = std::find(  //
        device.last_used_streams.begin(),
        device.last_used_streams.end(),
        stream_index
    );
    std::rotate(it, it + 1, device.last_used_streams.end());

    spdlog::debug("selected stream index {} for operation on GPU {}", stream_index, device_id);
    return device.streams[stream_index].get();
}

DeviceEvent DeviceResources::submit(
    DeviceId device_id,
    DeviceStreamSet stream_hint,
    DeviceEventSet deps,
    DeviceResourceOperation& op,
    std::vector<BufferAccessor> accessors
) {
    auto& state = *select_stream_for_operation(device_id, stream_hint, deps);

    try {
        GPUContextGuard guard {state.context};
        m_stream_manager->wait_for_events(state.stream, deps);

        op.execute(state.resource, std::move(accessors));

        m_stream_manager->wait_on_default_stream(state.stream);
        auto event = m_stream_manager->record_event(state.stream);

        state.last_event = event;
        return event;
    } catch (const std::exception& e) {
        try {
            m_stream_manager->wait_until_ready(state.stream);
        } catch (...) {
            KMM_PANIC_FMT("fatal error: {}", e.what());
        }

        throw;
    }
}

}  // namespace kmm
