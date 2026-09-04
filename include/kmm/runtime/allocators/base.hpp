#pragma once

#include <deque>
#include <limits>

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

    virtual AllocResult allocate_async(  //
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out
    );

    virtual void deallocate_async(  //
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout
    );

    virtual AllocResult allocate(BufferLayout layout, void** addr_out) = 0;

    virtual void deallocate(void* addr, BufferLayout layout) = 0;

    virtual void poll() {}

    virtual void trim(size_t nbytes_remaining) {}
};

}  // namespace kmm
