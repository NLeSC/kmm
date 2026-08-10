#pragma once

#include <cstddef>
#include <future>
#include <memory>
#include <vector>

#include "kmm/runtime/allocators/managed.hpp"
#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// Per-buffer counterpart to `MemorySystem`: knows the shape of a single buffer and how to
/// allocate, deallocate, and copy exactly that buffer at any of its possible locations.
class DataInterface {
  public:
    virtual ~DataInterface() = default;

    virtual size_t size_in_bytes() const = 0;

    /// Allocate this buffer on the given memory id. The returned dependencies indicate
    /// when the buffer is safe to be used.
    virtual AllocResult allocate(
        MemoryId memory_id,
        DeviceStream stream_hint,
        DeviceEventSet& deps_out
    ) = 0;

    /// Deallocate this buffer on the given memory id. The given dependencies indicate when
    /// the deallocation should happen.
    virtual void deallocate(
        MemoryId memory_id,
        DeviceStream stream_hint,
        const DeviceEventSet& deps
    ) = 0;

    /// Returns the pointer to the data for the given MemoryId. The caller must ensure that
    /// `allocate` was called before this function was called.
    virtual void* address(MemoryId memory_id) const = 0;

    /// Copy data from the given `src` to `dst` memory. The given dependencies indicate
    /// when the copy should be performed. The caller must have called `allocate` on both
    /// the `src` and `dst` in the past.
    virtual DeviceEvent copy(
        MemoryId src,
        MemoryId dst,
        DeviceStream stream_hint,
        const DeviceEventSet& deps
    ) = 0;

    /// Indicate if copying data from `src` to `dst` is supported.
    virtual bool is_copy_supported(MemoryId src, MemoryId dst) = 0;

    /// Hint that this buffer will be accessed on the given memory in the future when the given
    /// dependencies complete. This is just a hint, useful for `cudaMemPrefetchAsync`.
    virtual void hint_access(
        MemoryId memory_id,
        DeviceStream stream_hint,
        const DeviceEventSet& deps
    ) {}

    /// Initialize the buffer on the host. This returns a future as it is likely that this
    /// operation will be performed on the CPU.
    virtual std::future<void> initialize_host(const DeviceEventSet& deps) {
        return {};
    }

    /// Initialize the buffer on the GPU.
    virtual DeviceEvent initialize_device(
        DeviceId memory_id,
        DeviceStream stream_hint,
        const DeviceEventSet& deps
    ) {
        return DeviceEvent::null();
    }

    /// Allocate the buffer in `dst` memory and copy data immediately from `src` to `dst.
    /// This is typically just an alias for calling `allocate` followed by `copy`, but might
    /// be optimized
    virtual AllocResult allocate_and_copy(
        MemoryId src,
        MemoryId dst,
        DeviceStream stream_hint,
        const DeviceEventSet& deps_in,
        DeviceEvent& dep_out
    ) {
        DeviceEventSet deps;
        if (allocate(dst, stream_hint, deps) == AllocResult::Success) {
            deps.insert(deps_in);
            dep_out = copy(src, dst, stream_hint, deps);
            return AllocResult::Success;
        } else {
            return AllocResult::ErrorOutOfMemory;
        }
    }
};

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

    AllocResult allocate(MemoryId memory_id, DeviceStream stream_hint, DeviceEventSet& deps_out)
        override;
    void deallocate(MemoryId memory_id, DeviceStream stream_hint, const DeviceEventSet& deps)
        override;
    void* address(MemoryId memory_id) const override;

    DeviceEvent copy(
        MemoryId src,
        MemoryId dst,
        DeviceStream stream_hint,
        const DeviceEventSet& deps
    ) override;
    bool is_copy_supported(MemoryId src, MemoryId dst) override;

    std::future<void> initialize_host(const DeviceEventSet& deps) override;
    DeviceEvent initialize_device(
        DeviceId memory_id,
        DeviceStream stream_hint,
        const DeviceEventSet& deps
    ) override;

  private:
    BufferLayout m_layout;
    refcnt_ptr<MemorySystem> m_system;
    FillValue m_fill_value;
    void* m_host_ptr = nullptr;
    CUdeviceptr m_device_ptrs[MAX_DEVICES] {};
};

}  // namespace kmm
