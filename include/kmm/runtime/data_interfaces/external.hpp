#pragma once

#include "kmm/runtime/data_interfaces/base.hpp"

namespace kmm {

/// A `DataInterface` that wraps a pre-existing, externally-owned pointer pinned to a single
/// `MemoryId`. KMM never allocates, deallocates, or copies the underlying memory: `allocate`
/// and `deallocate` on the pinned memory are no-ops, and any attempt to access or copy the
/// buffer on a different `MemoryId` throws `std::runtime_error`.
class ExternalDataInterface final: public DataInterface {
  public:
    ExternalDataInterface(void* ptr, size_t size_in_bytes, MemoryId memory_id);

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

    void copy(
        MemoryId src,
        MemoryId dst,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEventSet& deps_out
    ) override;

    bool is_copy_supported(MemoryId src, MemoryId dst) const noexcept override;

  private:
    void check_memory_id(MemoryId memory_id) const;

    void* m_ptr;
    size_t m_size_in_bytes;
    MemoryId m_memory_id;
};

}  // namespace kmm
