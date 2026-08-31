#include <algorithm>
#include <utility>

#include "kmm/core/integer_fun.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interfaces/host.hpp"
#include "kmm/runtime/memops/fill.hpp"

namespace kmm {

static BufferLayout normalize_buffer_layout(BufferLayout layout) {
    static constexpr size_t max_align = 128;
    size_t k = std::min(std::max(layout.size_in_bytes, layout.alignment), max_align);
    size_t align = round_up_to_power_of_two(k);
    return {round_up_to_multiple(layout.size_in_bytes, align), align};
}

HostDataInterface::HostDataInterface(
    BufferLayout layout,
    refcnt_ptr<MemorySystem> system
) :
    m_layout(normalize_buffer_layout(layout)),
    m_system(std::move(system)) {}

size_t HostDataInterface::size_in_bytes() const {
    return m_layout.size_in_bytes;
}

AllocResult HostDataInterface::allocate(
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    DeviceEventSet& deps_out
) {
    if (m_refcount == 0) {
        DeviceEventSet deps;
        auto result = m_system->allocate_host(m_layout, &m_host_ptr, stream_hint, deps);

        if (result != AllocResult::Success) {
            return result;
        }

        m_alloc_deps = std::move(deps);
    }

    m_refcount++;
    deps_out.insert(m_alloc_deps);
    return AllocResult::Success;
}

void HostDataInterface::deallocate(
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    KMM_ASSERT(m_refcount > 0);
    m_dealloc_deps.insert(deps);

    if (--m_refcount == 0) {
        m_system->deallocate_host(m_host_ptr, m_layout, stream_hint, m_dealloc_deps);
        m_host_ptr = nullptr;
        m_alloc_deps.clear();
        m_dealloc_deps.clear();
    }
}

void* HostDataInterface::address(MemoryId memory_id) const {
    if (memory_id.is_host()) {
        return m_host_ptr;
    }

    return m_system->translate_host_pointer(memory_id.as_device(), m_host_ptr);
}

bool HostDataInterface::is_copy_supported(MemoryId src, MemoryId dst) {
    return m_system->is_copy_supported(src, dst);
}

void HostDataInterface::copy(
    MemoryId src,
    MemoryId dst,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps,
    DeviceEventSet& deps_out
) {
    deps_out.insert(deps);
}


}  // namespace kmm
