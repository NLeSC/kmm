#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class PinnedMemoryAllocator: public Allocator {
  public:
    PinnedMemoryAllocator(CUcontext context);

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    CUcontext m_context;
};

}  // namespace kmm
