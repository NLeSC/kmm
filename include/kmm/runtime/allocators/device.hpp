#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

struct Allocation {
    void* addr;
    size_t nbytes;
    DeviceEvent event;
};

class PinnedMemoryAllocator: public SyncAllocator {
  public:
    PinnedMemoryAllocator(
        GPUContextHandle context,
        std::shared_ptr<DeviceStreamManager> streams,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );

    AllocationResult allocate(size_t nbytes, void** addr_out) final;
    void deallocate(void* addr, size_t nbytes) final;

  private:
    GPUContextHandle m_context;
};

class DeviceMemoryAllocator: public SyncAllocator {
  public:
    DeviceMemoryAllocator(
        GPUContextHandle context,
        std::shared_ptr<DeviceStreamManager> streams,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );

    AllocationResult allocate(size_t nbytes, void** addr_out) final;
    void deallocate(void* addr, size_t nbytes) final;

  private:
    GPUContextHandle m_context;
};

enum struct DevicePoolKind { Default, Create };

class DevicePoolAllocator: public AsyncAllocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(DevicePoolAllocator)

  public:
    DevicePoolAllocator(
        GPUContextHandle context,
        std::shared_ptr<DeviceStreamManager> streams,
        DevicePoolKind kind = DevicePoolKind::Create,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );
    ~DevicePoolAllocator();
    AllocationResult allocate_async(size_t nbytes, void** addr_out, DeviceEventSet& deps_out) final;
    void deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) final;
    void make_progress() final;
    void trim(size_t nbytes_remaining = 0) final;

  private:
    GPUContextHandle m_context;
    g_memory_pool_t m_pool;
    std::shared_ptr<DeviceStreamManager> m_streams;
    DeviceStream m_alloc_stream;
    DeviceStream m_dealloc_stream;
    std::deque<Allocation> m_pending_deallocs;
    DevicePoolKind m_kind;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_limit;
};

}  // namespace kmm