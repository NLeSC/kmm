#include <utility>

#include "kmm/runtime/allocators/base.hpp"
#include "kmm/runtime/device_event.hpp"

namespace kmm {

SyncAllocator::SyncAllocator(size_t max_bytes) : m_bytes_limit(max_bytes) {}

SyncAllocator::~SyncAllocator() {
    // Force through any deallocations still waiting on their dependencies,
    // otherwise their memory would never actually be freed.
    trim(0);
}

AllocResult SyncAllocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out,
    DeviceEventSet& deps_out
) {
    if (layout.size_in_bytes > m_bytes_limit - m_bytes_in_use) {
        return AllocResult::ErrorOutOfMemory;
    }

    // we must wait for all events on the stream to finish
    stream.synchronize();

    AllocResult result = allocate(layout, addr_out);

    if (result == AllocResult::Success) {
        m_bytes_in_use += layout.size_in_bytes;
    }

    return result;
}

void SyncAllocator::deallocate_async(  //
    const DeviceStream& stream,
    void* addr,
    BufferLayout layout,
    DeviceEventSet deps
) {
    // we must wait for all events on the stream to finish
    stream.synchronize();

    for (const auto& e : deps) {
        e.synchronize();
    }

    m_bytes_in_use -= layout.size_in_bytes;
    deallocate(addr, layout);
}

}  // namespace kmm
