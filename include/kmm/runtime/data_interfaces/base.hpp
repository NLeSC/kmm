#pragma once

#include <future>

#include "kmm/runtime/allocators/base.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"

namespace kmm {

class MemorySystem;

/// Per-buffer counterpart to `MemorySystem`: knows the shape of a single buffer and how to
/// allocate, deallocate, and copy exactly that buffer at any of its possible locations. The
/// `MemorySystem` that backs the buffer is not held by the interface; the caller passes it into
/// every method that needs it.
///
/// Concrete implementations live in their own headers under `data_interfaces/`: `FlatDataInterface`
/// (flat.hpp), `ManagedDataInterface` (managed.hpp), `PinnedDataInterface` (pinned.hpp), and
/// `ExternalDataInterface` (external.hpp).
class DataInterface {
  public:
    virtual ~DataInterface() = default;

    virtual size_t size_in_bytes() const noexcept = 0;

    /// Allocate this buffer on the given memory id. The returned dependencies indicate
    /// when the buffer is safe to be used.
    virtual AllocResult allocate(
        MemorySystem& system,
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        DeviceEventSet& deps_out
    ) = 0;

    /// Deallocate this buffer on the given memory id. The given dependencies indicate when
    /// the deallocation should happen.
    virtual void deallocate(
        MemorySystem& system,
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) = 0;

    /// Returns the pointer to the data for the given MemoryId. The caller must ensure that
    /// `allocate` was called before this function was called.
    virtual void* address(MemoryId memory_id) const noexcept = 0;

    /// Copy data from the given `src` to `dst` memory. The given dependencies indicate
    /// when the copy should be performed. The caller must have called `allocate` on both
    /// the `src` and `dst` in the past.
    virtual void copy(
        MemorySystem& system,
        MemoryId src,
        MemoryId dst,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEventSet& deps_out
    ) = 0;

    /// Indicate if copying data from `src` to `dst` is supported.
    virtual bool is_copy_supported(
        MemorySystem& system,
        MemoryId src,
        MemoryId dst
    ) const noexcept = 0;

    /// Hint that this buffer will be accessed on the given memory in the future when the given
    /// dependencies complete. This is just a hint, useful for `cudaMemPrefetchAsync`.
    virtual void hint_access(
        MemorySystem& system,
        MemoryId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) {}

    /// Initialize the buffer on the host. This returns a future as it is likely that this
    /// operation will be performed on the CPU.
    virtual std::future<void> initialize_host(MemorySystem& system, const DeviceEventSet& deps) {
        return {};
    }

    /// Initialize the buffer on the GPU.
    virtual DeviceEvent initialize_device(
        MemorySystem& system,
        DeviceId memory_id,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps
    ) {
        return DeviceEvent::null();
    }

    /// Allocate the buffer in `dst` memory and copy data immediately from `src` to `dst.
    /// This is typically just an alias for calling `allocate` followed by `copy`, but might
    /// be optimized
    virtual AllocResult allocate_and_copy(
        MemorySystem& system,
        MemoryId src,
        MemoryId dst,
        const DeviceStreamId& stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEventSet& deps_out
    ) {
        DeviceEventSet deps;
        auto result = allocate(system, dst, stream_hint, deps);

        if (result == AllocResult::Success) {
            try {
                deps.insert(deps_in);
                copy(system, src, dst, stream_hint, deps, deps_out);
            } catch (...) {
                deallocate(system, dst, stream_hint, deps);
                throw;
            }
        }

        return result;
    }
};

}  // namespace kmm
