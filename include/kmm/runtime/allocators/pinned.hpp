#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class PinnedMemoryAllocator: public Allocator {
  public:
    PinnedMemoryAllocator(GPUContext context);

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    GPUContext m_context;
};

}  // namespace kmm
