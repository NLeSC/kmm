#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class DeviceMemoryAllocator: public SyncAllocator {
  public:
    DeviceMemoryAllocator(CUcontext context, size_t max_bytes = std::numeric_limits<size_t>::max());

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;
    void deallocate(void* addr, BufferLayout layout) override final;

  private:
    CUcontext m_context;
};

}  // namespace kmm
