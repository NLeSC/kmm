#pragma once

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/data_interfaces/base.hpp"
#include "kmm/runtime/device_event_registry.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// A `DataInterface` that always lives in a single pinned host allocation
class HostDataInterface final: public DataInterface {
  public:
    /// If `fill_value` is non-empty, the buffer is filled with copies of it the first time it is
    /// materialized (see `initialize_host`/`initialize_device`).
    HostDataInterface(
        BufferLayout layout,
        refcnt_ptr<MemorySystem> system
    );

    size_t size_in_bytes() const override;

    AllocResult allocate(  //
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        DeviceEventSet& deps_out
    ) override;

    void deallocate(  //
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) override;

    void* address(  //
        MemoryId memory_id
    ) const override;

    bool is_copy_supported(MemoryId src, MemoryId dst) override;

    void copy(
        MemoryId src,
        MemoryId dst,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEventSet& deps_out
    ) override;

  private:
    BufferLayout m_layout;
    refcnt_ptr<MemorySystem> m_system;
    void* m_host_ptr = nullptr;
    size_t m_refcount = 0;
    DeviceEventSet m_alloc_deps;
    DeviceEventSet m_dealloc_deps;
};

}  // namespace kmm
