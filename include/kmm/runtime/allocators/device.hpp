#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class DeviceMemoryAllocator: public Allocator {
  public:
    DeviceMemoryAllocator(CUcontext context);

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    CUcontext m_context;
};

}  // namespace kmm
