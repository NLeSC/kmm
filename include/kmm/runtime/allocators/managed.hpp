#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class ManagedMemoryAllocator: Allocator {
  public:
    ManagedMemoryAllocator(
        CUcontext context
    );

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    CUcontext m_context;
};

}  // namespace kmm
