#pragma once

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/data_interfaces/base.hpp"
#include "kmm/runtime/device_event_registry.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// A `DataInterface` backed by a single pinned host allocation. Devices access it directly
/// (zero-copy) through a mapped pointer, so `copy` is a no-op and no device-side copy is ever
/// staged.
class PinnedDataInterface final: public DataInterface {
  public:
    PinnedDataInterface(
        BufferLayout layout,
        refcnt_ptr<MemorySystem> system
    );

    size_t size_in_bytes() const noexcept override;

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
    ) const noexcept override;

    bool is_copy_supported(MemoryId src, MemoryId dst) const noexcept override;

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
    void* m_device_ptrs[MAX_DEVICES] {};
    size_t m_refcount = 0;
    DeviceEventSet m_alloc_deps;
    DeviceEventSet m_dealloc_deps;
};

}  // namespace kmm
