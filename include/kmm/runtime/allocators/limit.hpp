#pragma once

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class LimitAllocator: public Allocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(LimitAllocator)

    struct Allocation {
        void* addr;
        size_t nbytes;
        DeviceEvent event;
    };

  public:
    LimitAllocator(std::unique_ptr<Allocator> inner, DeviceEventRegistry events, size_t max_size);
    ~LimitAllocator();

    AllocResult allocate_async(
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out
    ) override final;

    void deallocate_async(  //
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout
    ) override final;

    AllocResult allocate(BufferLayout layout, void** addr_out) override final;

    void deallocate(void* addr, BufferLayout layout) override final;

    void poll() final;

    void trim(size_t nbytes_remaining) final;

  private:
    bool ensure_enough_space(const DeviceStream* stream, size_t nbytes);

    std::unique_ptr<Allocator> m_inner;
    DeviceEventRegistry m_events;
    std::deque<Allocation> m_pending_deallocs;
    DeviceEventSet m_limit_barrier;

    // maximum number of bytes that can be allocated
    size_t m_bytes_limit = 0;

    // number of bytes in use. For these, allocate has been called but not deallocate yet.
    size_t m_bytes_active = 0;

    // number of bytes awaiting deallocation. For these, deallocate hase been called and
    // the deallocation event is current waiting in m_pending_deallocs.
    size_t m_bytes_pending = 0;
};

}  // namespace kmm