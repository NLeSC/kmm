#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class PinnedMemoryAllocator: public Allocator {
  public:
    PinnedMemoryAllocator(g_context_t context);

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    g_context_t m_context;
};

}  // namespace kmm
