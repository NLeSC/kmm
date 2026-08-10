#pragma once

#include <array>
#include <limits>
#include <memory>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/allocators/device.hpp"
#include "kmm/runtime/allocators/pinned.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream_registry.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/system_info.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// Owns the physical memory backing every buffer, and the means of moving bytes between host
/// and device memory. Host allocations come from a single portable pinned-memory pool
/// (`PinnedMemoryAllocator`), device allocations from one `cuMemAlloc`-backed pool per device
/// (`DeviceMemoryAllocator`), and every copy is issued asynchronously on a dedicated per-device
/// stream.
class MemorySystem: public reference_count<MemorySystem> {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemorySystem)

  public:
    MemorySystem(
        const SystemInfo& system_info,
        DeviceStreamRegistry* streams,
        size_t host_memory_limit = std::numeric_limits<size_t>::max(),
        size_t device_memory_limit = std::numeric_limits<size_t>::max()
    );

    ~MemorySystem();

    void make_progress();
    void trim_host(size_t bytes_remaining = 0);
    void trim_device(size_t bytes_remaining = 0);

    AllocResult allocate_host(
        size_t nbytes,
        DeviceId device_affinity,
        void** ptr_out,
        DeviceEventSet& deps_out
    );

    void deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps);

    AllocResult allocate_device(
        DeviceId device_id,
        size_t nbytes,
        CUdeviceptr* ptr_out,
        DeviceEventSet& deps_out
    );

    void deallocate_device(DeviceId device_id, CUdeviceptr ptr, size_t nbytes, DeviceEventSet deps);

    DeviceEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        CUdeviceptr dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    );

    DeviceEvent copy_device_to_host(
        DeviceId device_id,
        CUdeviceptr src_addr,
        void* dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    );

    DeviceEvent copy_device_to_device(
        DeviceId src_device,
        DeviceId dst_device,
        CUdeviceptr src_addr,
        CUdeviceptr dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    );

    /// True if a direct copy between `src` and `dst` is possible: always true when either side
    /// is host memory, and true between two devices only if peer access between them is
    /// available (queried once at construction).
    bool is_copy_supported(MemoryId src, MemoryId dst);

    /// The stream used for every allocation/copy issued against `device` (see `DeviceState`).
    /// Exposed so other device-side operations (e.g. `ReductionManager`'s `reduce_async` calls)
    /// can be sequenced against the same stream instead of needing one of their own.
    DeviceStream stream(DeviceId device) const;

  private:
    struct DeviceState;

    DeviceState& device_state(DeviceId id) const;

    std::shared_ptr<DeviceStreamRegistry> m_streams;
    std::unique_ptr<AsyncAllocator> m_host_allocator;
    std::array<std::unique_ptr<DeviceState>, MAX_DEVICES> m_devices;
    size_t m_num_devices = 0;
    bool m_peer_access[MAX_DEVICES][MAX_DEVICES] {};
};

}  // namespace kmm
