#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/device_event_registry.hpp"
#include "kmm/runtime/device_stream.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/system_info.hpp"

namespace kmm {

enum class StreamKind : uint8_t { DeviceToDevice = 0, HostToDevice = 1, DeviceToHost = 2 };

std::ostream& operator<<(std::ostream&, StreamKind kind);

class DeviceDataStreams {
  public:
    DeviceDataStreams(
        const SystemInfo& info,
        DeviceEventRegistry events,
        size_t num_d2d_streams,
        size_t num_h2d_streams,
        size_t num_d2h_streams
    );

    DeviceDataStreams(DeviceDataStreams&&) noexcept;
    ~DeviceDataStreams();

    /**
     * Acquire a free stream of the given kind on `device_id`, preferring the one predicted to
     * become ready soonest for work depending on `deps`.
     */
    DeviceStreamId acquire_stream(DeviceId device_id, StreamKind kind, const DeviceEventSet& deps);

    /**
     * See `release_stream(const DeviceStream&, uint64_t)`. Useful when only the id of the stream
     * to release is known, not the `DeviceStream` object itself.
     */
    DeviceEvent release_stream(DeviceStreamId stream_id, uint64_t cost = 1);

    /**
     * Convenience method that acquires a stream of the given kind on `device_id`, calls
     * `fun(stream)` to submit work onto it, and then releases the stream again.
     */
    template<typename F>
    DeviceEvent submit(DeviceId device_id, StreamKind kind, const DeviceEventSet& deps, F&& fun) {
        auto stream_id = acquire_stream(device_id, kind, deps);
        uint64_t cost;

        try {
            cost = std::forward<F>(fun)(stream_id);
        } catch (...) {
            release_stream(stream_id);
            throw;
        }

        return release_stream(stream_id, cost);
    }

    void make_progress();

    struct Impl;

  private:
    std::unique_ptr<Impl> m_impl;
};

}  // namespace kmm

template<>
struct fmt::formatter<kmm::StreamKind>: fmt::ostream_formatter {};