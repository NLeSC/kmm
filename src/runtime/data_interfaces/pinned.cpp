#include <algorithm>
#include <utility>

#include "kmm/core/integer_fun.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interfaces/pinned.hpp"
#include "kmm/runtime/memops/fill.hpp"

namespace kmm {

static BufferLayout normalize_buffer_layout(BufferLayout layout) {
    static constexpr size_t max_align = 128;
    size_t k = std::min(std::max(layout.size_in_bytes, layout.alignment), max_align);
    size_t align = round_up_to_power_of_two(k);
    return {round_up_to_multiple(layout.size_in_bytes, align), align};
}

PinnedDataInterface::PinnedDataInterface(BufferLayout layout, refcnt_ptr<MemorySystem> system) :
    m_layout(normalize_buffer_layout(layout)),
    m_system(std::move(system)) {}

size_t PinnedDataInterface::size_in_bytes() const noexcept {
    return m_layout.size_in_bytes;
}

AllocResult PinnedDataInterface::allocate(
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

    // Devices reach the single host allocation through a mapped pointer. Resolve here now so
    // `allocate` throws the exception and address remains exception-free.
    if (!memory_id.is_host()) {
        auto device_id = memory_id.as_device();
        m_device_ptrs[device_id.get()] = m_system->translate_host_pointer(device_id, m_host_ptr);
    }

    m_refcount++;
    deps_out.insert(m_alloc_deps);
    return AllocResult::Success;
}

void PinnedDataInterface::deallocate(
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    KMM_ASSERT(m_refcount > 0);
    m_dealloc_deps.insert(deps);

    if (--m_refcount == 0) {
        m_system->deallocate_host(m_host_ptr, m_layout, stream_hint, m_dealloc_deps);
        m_host_ptr = nullptr;
        for (auto& ptr : m_device_ptrs) {
            ptr = nullptr;
        }
        m_alloc_deps.clear();
        m_dealloc_deps.clear();
    }
}

void* PinnedDataInterface::address(MemoryId memory_id) const noexcept {
    if (memory_id.is_host()) {
        return m_host_ptr;
    }

    return m_device_ptrs[memory_id.as_device().get()];
}

bool PinnedDataInterface::is_copy_supported(MemoryId src, MemoryId dst) const noexcept {
    return m_system->is_copy_supported(src, dst);
}

void PinnedDataInterface::copy(
    MemoryId src,
    MemoryId dst,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps,
    DeviceEventSet& deps_out
) {
    deps_out.insert(deps);
}

}  // namespace kmm
