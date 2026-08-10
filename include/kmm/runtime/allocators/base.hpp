#pragma once

#include <deque>
#include <limits>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream_registry.hpp"

namespace kmm {

enum struct AllocResult { Success, ErrorOutOfMemory };

class AsyncAllocator {
  public:
    virtual ~AsyncAllocator() = default;

    virtual AllocResult allocate_async(
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out,
        DeviceEventSet& deps_out
    ) = 0;

    virtual void deallocate_async(  //
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout,
        DeviceEventSet deps
    ) = 0;

    virtual void poll() {}

    virtual void trim(size_t nbytes_remaining) {}
};

class SyncAllocator: public AsyncAllocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(SyncAllocator)

  public:
    SyncAllocator(size_t max_bytes = std::numeric_limits<size_t>::max());
    ~SyncAllocator() override;

    virtual AllocResult allocate(BufferLayout layout, void** addr_out) = 0;

    virtual void deallocate(void* addr, BufferLayout layout) = 0;

    AllocResult allocate_async(  //
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out,
        DeviceEventSet& deps_out
    ) override final;

    void deallocate_async(  //
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout,
        DeviceEventSet deps
    ) override final;

  private:
    size_t m_bytes_limit;
    size_t m_bytes_in_use = 0;
};

}  // namespace kmm
