#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class ManagedMemoryAllocator: public SyncAllocator {
  public:
    ManagedMemoryAllocator(
        CUcontext context,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    CUcontext m_context;
};

}  // namespace kmm
