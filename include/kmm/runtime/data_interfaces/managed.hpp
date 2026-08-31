#pragma once

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/data_interfaces/base.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// A `DataInterface` backed by a single CUDA/HIP managed-memory allocation (`cudaMallocManaged`),
/// resident on the host and every device simultaneously. There is only ever one physical
/// allocation: `allocate`/`deallocate` are refcounted no-ops beyond the first/last call
/// (regardless of which `MemoryId` is requested), `address` returns the same pointer for every
/// `MemoryId`, and `copy` is a no-op since the driver keeps the allocation coherent everywhere.
class ManagedDataInterface final: public DataInterface {
  public:
    /// If `fill_value` is non-empty, the buffer is filled with copies of it the first time it is
    /// materialized (see `initialize_host`/`initialize_device`).
    ManagedDataInterface(
        BufferLayout layout,
        refcnt_ptr<MemorySystem> system,
        FillValue fill_value = {}
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

    /// Prefetches the allocation to `memory_id` (`cudaMemPrefetchAsync`/`hipMemPrefetchAsync`) so
    /// the driver can start migrating pages before the actual access. Best-effort: fired on the
    /// default stream without waiting on `deps`, since a prefetch racing with a pending write is
    /// merely a missed optimization (the driver still page-faults correctly), not a correctness
    /// issue.
    void hint_access(
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) override;

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

  private:
    BufferLayout m_layout;
    refcnt_ptr<MemorySystem> m_system;
    FillValue m_fill_value;
    void* m_ptr = nullptr;
    size_t m_refcount = 0;
    DeviceEventSet m_alloc_deps;
    DeviceEventSet m_dealloc_deps;
};

}  // namespace kmm
