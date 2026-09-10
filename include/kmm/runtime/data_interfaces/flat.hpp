#pragma once

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/data_interfaces/base.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

/// Default `DataInterface`: a flat buffer of `layout.size_in_bytes` bytes, allocated and copied
/// through a shared `MemorySystem`. Equivalent to how every buffer behaved before per-buffer
/// `DataInterface`s existed.
class FlatDataInterface final: public DataInterface {
  public:
    /// If `fill_value` is non-empty, the buffer is filled with copies of it the first time it is
    /// materialized in any memory (see `initialize_host`/`initialize_device`).
    FlatDataInterface(BufferLayout layout, FillValue fill_value = {});

    size_t size_in_bytes() const noexcept override;

    AllocResult allocate(  //
        MemorySystem& system,
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        DeviceEventSet& deps_out
    ) override;

    void deallocate(  //
        MemorySystem& system,
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) override;

    void* address(  //
        MemoryId memory_id
    ) const noexcept override;

    bool is_copy_supported(
        MemorySystem& system,
        MemoryId src,
        MemoryId dst
    ) const noexcept override;

    void copy(
        MemorySystem& system,
        MemoryId src,
        MemoryId dst,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEventSet& deps_out
    ) override;

    std::future<void> initialize_host(MemorySystem& system, const DeviceEventSet& deps) override;

    DeviceEvent initialize_device(
        MemorySystem& system,
        DeviceId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) override;

    AllocResult allocate_and_copy(
        MemorySystem& system,
        MemoryId src,
        MemoryId dst,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEventSet& deps_out
    ) override;

  private:
    BufferLayout m_layout;
    FillValue m_fill_value;
    void* m_host_ptr = nullptr;
    g_device_ptr_t m_device_ptrs[MAX_DEVICES] {};
};

}  // namespace kmm
