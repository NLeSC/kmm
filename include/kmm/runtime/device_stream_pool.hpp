#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream.hpp"
#include "kmm/runtime/device_stream_registry.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/system_info.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

enum class StreamKind { Compute, CopyH2D, copyD2H };

/// Owns a fixed-size set of `DeviceStream`s per device per `StreamKind`, all created once at
/// construction (CUDA streams are cheap to reuse but not free to create, so none are made on
/// demand). `get_stream` hands back one of them on every call.
class DeviceStreamPool {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceStreamPool)

  public:
    DeviceStreamPool(
        const SystemInfo& system_info,
        DeviceStreamRegistry& streams,
        size_t num_data_streams = 2,
        size_t num_compute_streams = 4
    );

    ~DeviceStreamPool();

    DeviceStream get_stream(DeviceId device, StreamKind kind, const DeviceEventSet& deps = {})
        const;

  private:
    struct PooledStream {
        PooledStream(DeviceStream stream) : stream(std::move(stream)) {}

        PooledStream(PooledStream&& that) noexcept :
            stream(std::move(that.stream)),
            last_kind(that.last_kind.load(std::memory_order_relaxed)) {}

        DeviceStream stream;
        mutable std::atomic<StreamKind> last_kind {StreamKind::Compute};
    };

    struct DeviceStreams {
        std::vector<PooledStream> data;
        std::vector<PooledStream> compute;
        mutable std::atomic<size_t> next_data {0};
        mutable std::atomic<size_t> next_compute {0};
    };

    static const PooledStream* pick(
        const std::vector<PooledStream>& pool,
        std::atomic<size_t>& cursor,
        StreamKind kind,
        const DeviceEventSet& deps,
        bool match_direction
    );

    std::array<DeviceStreams, MAX_DEVICES> m_devices;
    size_t m_num_devices = 0;
};

}  // namespace kmm
