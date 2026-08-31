#pragma once

#include "kmm/runtime/allocators/managed.hpp"
#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/data_interfaces/base.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// Default `DataInterface`: a flat buffer of `layout.size_in_bytes` bytes, allocated and copied
/// through a shared `MemorySystem`. Equivalent to how every buffer behaved before per-buffer
/// `DataInterface`s existed.
class FlatDataInterface final: public DataInterface {
  public:
    /// If `fill_value` is non-empty, the buffer is filled with copies of it the first time it is
    /// materialized in any memory (see `initialize_host`/`initialize_device`).
    FlatDataInterface(
        BufferLayout layout,
        refcnt_ptr<MemorySystem> system,
        FillValue fill_value = {}
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

    std::future<void> initialize_host(const DeviceEventSet& deps) override;

    DeviceEvent initialize_device(
        DeviceId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) override;

    AllocResult allocate_and_copy(
        MemoryId src,
        MemoryId dst,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEventSet& deps_out
    ) override;

  private:
    BufferLayout m_layout;
    refcnt_ptr<MemorySystem> m_system;
    FillValue m_fill_value;
    void* m_host_ptr = nullptr;
    GPUDeviceptr m_device_ptrs[MAX_DEVICES] {};
};

}  // namespace kmm
