#pragma once

#include <deque>

#include "kmm/runtime/allocators/base.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

enum struct DevicePoolKind { Default, Create };

class DevicePoolAllocator: public AsyncAllocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(DevicePoolAllocator)

    struct Allocation {
        void* addr;
        size_t nbytes;
        DeviceEvent event;
    };

  public:
    DevicePoolAllocator(
        CUcontext context,
        DevicePoolKind kind = DevicePoolKind::Create,
        size_t max_size = std::numeric_limits<size_t>::max()
    );
    ~DevicePoolAllocator();

    AllocResult allocate_async(
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out,
        DeviceEventSet& deps_out
    ) override final;

    void deallocate_async(
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout,
        DeviceEventSet deps
    ) override final;

    void poll() override final;
    void trim(size_t nbytes_remaining) override final;

  private:
    bool is_allocation_allowed(size_t nbytes) const;
    bool ensure_enough_space(const DeviceStream& stream, size_t nbytes);

    CUcontext m_context;
    CUmemoryPool m_pool;
    std::deque<Allocation> m_pending_deallocs;
    DevicePoolKind m_kind;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_limit = 0;
};

}  // namespace kmm
