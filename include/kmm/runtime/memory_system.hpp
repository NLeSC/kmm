#pragma once

#include <algorithm>
#include <array>
#include <future>
#include <limits>
#include <memory>

#include "runtime_config.hpp"

#include "kmm/core/macros.hpp"
#include "kmm/runtime/allocators/device.hpp"
#include "kmm/runtime/allocators/pinned.hpp"
#include "kmm/runtime/device_data_streams.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/runtime/system_info.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

struct MemoryStats {
    size_t bytes_inuse = 0;
    size_t bytes_allocated = 0;
    size_t max_bytes_inuse = 0;
    size_t bytes_to_host = 0;
    size_t bytes_to_device[MAX_DEVICES] {};

    void record_allocation(size_t nbytes) {
        bytes_allocated += nbytes;
        bytes_inuse += nbytes;
        max_bytes_inuse = std::max(max_bytes_inuse, bytes_inuse);
    }

    void record_deallocation(size_t nbytes) {
        bytes_inuse -= nbytes;
    }
};

class MemorySystem: public reference_count<MemorySystem> {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemorySystem)

  public:
    MemorySystem(
        const SystemInfo& system_info,
        DeviceEventRegistry events,
        const RuntimeConfig& config
    );

    ~MemorySystem();

    void make_progress();
    void trim_host(size_t bytes_remaining = 0);
    void trim_device(size_t bytes_remaining = 0);

    AllocResult allocate_host(
        BufferLayout layout,
        void** ptr_out,
        const DeviceStreamId& stream_hint,
        DeviceEventSet& deps_out
    );

    void deallocate_host(
        void* ptr,
        BufferLayout layout,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    void* translate_host_pointer(DeviceId device_id, void* host_ptr) const;

    AllocResult allocate_managed(
        BufferLayout layout,
        void** ptr_out,
        const DeviceStreamId& stream_hint,
        DeviceEventSet& deps_out
    );

    void deallocate_managed(
        void* ptr,
        BufferLayout layout,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    void prefetch_managed(
        MemoryId memory_id,
        void* ptr,
        BufferLayout layout,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    );

    AllocResult allocate_device(
        DeviceId device_id,
        BufferLayout layout,
        g_device_ptr_t* ptr_out,
        const DeviceStreamId& stream_hint,
        DeviceEventSet& deps_out
    );

    void deallocate_device(
        DeviceId device_id,
        g_device_ptr_t ptr,
        BufferLayout layout,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    DeviceEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        g_device_ptr_t dst_addr,
        size_t nbytes,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    DeviceEvent copy_device_to_host(
        DeviceId device_id,
        g_device_ptr_t src_addr,
        void* dst_addr,
        size_t nbytes,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    DeviceEvent copy_device_to_device(
        DeviceId src_device,
        DeviceId dst_device,
        g_device_ptr_t src_addr,
        g_device_ptr_t dst_addr,
        size_t nbytes,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    AllocResult allocate_host_and_copy_from_device(
        BufferLayout layout,
        void** dst_addr,
        DeviceId device_id,
        g_device_ptr_t src_addr,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEvent& dep_out
    );

    AllocResult allocate_device_and_copy_from_host(
        DeviceId device_id,
        BufferLayout layout,
        g_device_ptr_t* dst_addr,
        const void* src_addr,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEvent& dep_out
    );

    DeviceEvent fill_device(
        DeviceId device_id,
        g_device_ptr_t addr,
        const FillDescription& description,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    /// Fill host memory at `addr` according to `description`. Runs on a background thread since
    /// this is a CPU-bound operation; the returned future becomes ready once `deps_in` have
    /// completed and the fill has finished.
    std::future<void> fill_host(
        void* addr,
        const FillDescription& description,
        const DeviceEventSet& deps_in
    );

    DeviceEvent reduce_device(
        DeviceId device_id,
        g_device_ptr_t src_addr,
        g_device_ptr_t dst_addr,
        const ReductionDescription& description,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in
    );

    bool is_copy_supported(MemoryId src, MemoryId dst);

  private:
    struct DeviceState;

    DeviceState& device_state(DeviceId id) const;
    DeviceId affinity_for_stream(const DeviceStreamId& stream_hint);
    bool same_context(DeviceId device_id, const DeviceStreamId& stream_hint);

    DeviceEventRegistry m_events;
    DeviceDataStreams m_streams;
    std::unique_ptr<Allocator> m_host_allocator;
    MemoryStats m_host_stats;
    std::unique_ptr<Allocator> m_managed_allocator;
    MemoryStats m_managed_stats;
    std::array<std::unique_ptr<DeviceState>, MAX_DEVICES> m_devices;
    size_t m_num_devices = 0;
    bool m_peer_access[MAX_DEVICES][MAX_DEVICES] {};
};

}  // namespace kmm
