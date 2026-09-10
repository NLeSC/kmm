#include <utility>

#include "kmm/runtime/allocators/base.hpp"
#include "kmm/runtime/device_event.hpp"

namespace kmm {

Allocator::Allocator() = default;

Allocator::~Allocator() {
    // Force through any deallocations still waiting on their dependencies,
    // otherwise their memory would never actually be freed.
    trim(0);
}

AllocResult Allocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out
) {
    // we must wait for all events on the stream to finish
    stream.synchronize();
    return allocate(layout, addr_out);
}

void Allocator::deallocate_async(  //
    const DeviceStream& stream,
    void* addr,
    BufferLayout layout
) {
    // we must wait for all events on the stream to finish
    stream.synchronize();
    deallocate(addr, layout);
}

}  // namespace kmm
