#pragma once

#include <deque>
#include <limits>
#include <optional>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream.hpp"

namespace kmm {

enum struct AllocResult { Success, ErrorOutOfMemory, ErrorUnsupported, ErrorPending };

class Allocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(Allocator)

  public:
    Allocator();
    virtual ~Allocator();

    /// allocate memory asynchronously on the given stream.
    virtual AllocResult allocate_async(  //
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out
    );

    /// deallocate memory asynchronously on the given stream. The memory must previously be allocated
    /// using `allocate_async` or `allocate`.
    virtual void deallocate_async(  //
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout
    );

    /// allocate memory synchronously, potentially blocking until memory becomes available.
    virtual AllocResult allocate(BufferLayout layout, void** addr_out) = 0;

    /// deallocate memory synchronously. The memory must previously be allocated using `allocate_async` or `allocate`.
    virtual void deallocate(void* addr, BufferLayout layout) = 0;

    /// Called many times per second, allowing the allocator to update internal bookkeeping.
    virtual void poll() {}

    /// Reduce the number of bytes this allocator holds reserved from the OS/driver to the given limit. For example,
    /// for a block/pool allocator, this will free unused blocks until only the given number of bytes remain.
    /// Note that this method is a hint as allocators may sometimes not be able to trim to the given limit.
    virtual void trim(size_t nbytes_remaining) {}

    /// Real number of bytes this allocator holds reserved from the OS/driver (which, for a
    /// block/pool allocator, is much larger than the sum of the live allocation sizes).
    /// `std::nullopt` if the allocator does not track this.
    virtual std::optional<size_t> bytes_reserved() const {
        return std::nullopt;
    }
};

}  // namespace kmm
