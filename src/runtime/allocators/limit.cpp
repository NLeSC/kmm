#include <cstring>
#include <utility>

#include "kmm/runtime/allocators/limit.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

LimitAllocator::LimitAllocator(
    std::unique_ptr<Allocator> inner,
    DeviceEventRegistry events,
    size_t max_size
) :
    m_inner(std::move(inner)),
    m_events(std::move(events)),
    m_bytes_limit(max_size),
    m_bytes_active(0),
    m_bytes_pending(0) {}

LimitAllocator::~LimitAllocator() {
    // wait until all evenst are done
    while (!m_pending_deallocs.empty()) {
        auto front = m_pending_deallocs.front();
        m_events.synchronize(front.event);
        m_bytes_pending -= front.nbytes;
        m_pending_deallocs.pop_front();
    }

    KMM_ASSERT(m_bytes_active == 0 && m_bytes_pending == 0);
}

AllocResult LimitAllocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out
) {
    size_t nbytes = layout.size_in_bytes;
    poll();

    auto deps = DeviceEventSet {};

    if (!ensure_enough_space(&stream, nbytes)) {
        return AllocResult::ErrorOutOfMemory;
    }

    // the stream must wait until reaching the barrier
    stream.wait_on_event(m_limit_barrier);

    auto result = m_inner->allocate_async(stream, layout, addr_out);

    if (result != AllocResult::Success) {
        return result;
    }

    m_bytes_active += nbytes;
    return AllocResult::Success;
}

void LimitAllocator::deallocate_async(const DeviceStream& stream, void* addr, BufferLayout layout) {
    m_inner->deallocate_async(stream, addr, layout);
    auto event = stream.record_event();
    m_bytes_active -= layout.size_in_bytes;
    m_bytes_pending += layout.size_in_bytes;
    m_pending_deallocs.push_back({addr, layout.size_in_bytes, event});
}

AllocResult LimitAllocator::allocate(BufferLayout layout, void** addr_out) {
    size_t nbytes = layout.size_in_bytes;
    poll();

    auto deps = DeviceEventSet {};

    if (!ensure_enough_space(nullptr, nbytes)) {
        return AllocResult::ErrorOutOfMemory;
    }

    // the stream must wait until reaching the barrier
    for (const auto& dep : m_limit_barrier) {
        m_events.synchronize(dep);
    }

    auto result = m_inner->allocate(layout, addr_out);

    if (result != AllocResult::Success) {
        return result;
    }

    m_bytes_active += nbytes;
    return AllocResult::Success;
}

void LimitAllocator::deallocate(void* addr, BufferLayout layout) {
    m_inner->deallocate(addr, layout);
    m_bytes_active -= layout.size_in_bytes;
}

void LimitAllocator::poll() {
    while (!m_pending_deallocs.empty()) {
        auto front = m_pending_deallocs.front();

        if (!m_events.is_ready(front.event)) {
            break;
        }

        m_bytes_pending -= front.nbytes;
        m_pending_deallocs.pop_front();
    }

    m_limit_barrier.prune(m_events);
    m_inner->poll();
}

void LimitAllocator::trim(size_t nbytes_remaining) {
    while (m_bytes_active + m_bytes_pending > nbytes_remaining) {
        if (m_pending_deallocs.empty()) {
            break;
        }

        auto& d = m_pending_deallocs.front();
        m_events.synchronize(d.event);

        m_bytes_pending -= d.nbytes;
        m_pending_deallocs.pop_front();
    }

    m_inner->trim(nbytes_remaining);
}

bool LimitAllocator::ensure_enough_space(const DeviceStream* stream, size_t nbytes) {
    // we can allocate now, we are done
    size_t remaining_bytes = m_bytes_limit - m_bytes_active - m_bytes_pending;
    if (remaining_bytes >= nbytes) {
        return true;
    }

    // even freeing everything pending would not be enough, no point waiting on any of it
    if (m_bytes_limit - m_bytes_active < nbytes) {
        return false;
    }

    auto it = m_pending_deallocs.begin();

    while (it != m_pending_deallocs.end()) {
        m_limit_barrier.insert(it->event);
        m_bytes_pending -= it->nbytes;
        remaining_bytes += it->nbytes;
        it = m_pending_deallocs.erase(it);

        if (remaining_bytes >= nbytes) {
            return true;
        }
    }

    return false;
}

}  // namespace kmm
