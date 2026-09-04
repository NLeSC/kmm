#pragma once

#include <deque>

#include "kmm/runtime/allocators/base.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

enum struct DevicePoolKind { Default, Create };

class DevicePoolAllocator: public Allocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(DevicePoolAllocator)

  public:
    DevicePoolAllocator(
        g_context_t context,
        DevicePoolKind kind = DevicePoolKind::Create,
        size_t max_size = std::numeric_limits<size_t>::max()
    );
    ~DevicePoolAllocator();

    AllocResult allocate_async(
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out
    ) override final;

    void deallocate_async(
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout
    ) override final;

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;

    void deallocate(void* addr, BufferLayout layout) override final;

    void trim(size_t nbytes_remaining) override final;

  private:
    g_context_t m_context;
    g_memory_pool_t m_pool;
    DevicePoolKind m_kind;
};

}  // namespace kmm
