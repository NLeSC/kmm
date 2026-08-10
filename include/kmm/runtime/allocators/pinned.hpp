#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class PinnedMemoryAllocator: public SyncAllocator {
  public:
    PinnedMemoryAllocator(CUcontext context, size_t max_bytes = std::numeric_limits<size_t>::max());

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    CUcontext m_context;
};

}  // namespace kmm
