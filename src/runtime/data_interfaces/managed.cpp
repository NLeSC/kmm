#include <algorithm>
#include <utility>

#include "kmm/core/integer_fun.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interfaces/managed.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

static BufferLayout normalize_buffer_layout(BufferLayout layout) {
    static constexpr size_t max_align = 128;
    size_t k = std::min(std::max(layout.size_in_bytes, layout.alignment), max_align);
    size_t align = round_up_to_power_of_two(k);
    return {round_up_to_multiple(layout.size_in_bytes, align), align};
}

ManagedDataInterface::ManagedDataInterface(BufferLayout layout, FillValue fill_value) :
    m_layout(normalize_buffer_layout(layout)),
    m_fill_value(std::move(fill_value)) {}

size_t ManagedDataInterface::size_in_bytes() const noexcept {
    return m_layout.size_in_bytes;
}

AllocResult ManagedDataInterface::allocate(
    MemorySystem& system,
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    DeviceEventSet& deps_out
) {
    if (m_refcount == 0) {
        DeviceEventSet deps;
        auto result = system.allocate_managed(m_layout, &m_ptr, stream_hint, deps);

        if (result != AllocResult::Success) {
            return result;
        }

        m_alloc_deps = std::move(deps);
    }

    m_refcount++;
    deps_out.insert(m_alloc_deps);
    return AllocResult::Success;
}

void ManagedDataInterface::deallocate(
    MemorySystem& system,
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    KMM_ASSERT(m_refcount > 0);
    m_dealloc_deps.insert(deps);

    if (--m_refcount == 0) {
        system.deallocate_managed(m_ptr, m_layout, stream_hint, m_dealloc_deps);
        m_ptr = nullptr;
        m_alloc_deps.clear();
        m_dealloc_deps.clear();
    }
}

void* ManagedDataInterface::address(MemoryId memory_id) const noexcept {
    return m_ptr;
}

bool ManagedDataInterface::is_copy_supported(
    MemorySystem& system,
    MemoryId src,
    MemoryId dst
) const noexcept {
    return true;
}

void ManagedDataInterface::hint_access(
    MemorySystem& system,
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    system.prefetch_managed(memory_id, m_ptr, m_layout, stream_hint, deps);
}

void ManagedDataInterface::copy(
    MemorySystem& system,
    MemoryId src,
    MemoryId dst,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps,
    DeviceEventSet& deps_out
) {
    deps_out.insert(deps);
}

std::future<void> ManagedDataInterface::initialize_host(
    MemorySystem& system,
    const DeviceEventSet& deps
) {
    if (m_fill_value.length == 0) {
        return {};
    }

    size_t element_size = m_fill_value.length;
    size_t count = m_layout.size_in_bytes / element_size;

    FillDescription description(m_fill_value);
    description.add_dimension(
        static_cast<memops_extent_type>(count),
        static_cast<memops_stride_type>(element_size)
    );

    return system.fill_host(m_ptr, description, deps);
}

DeviceEvent ManagedDataInterface::initialize_device(
    MemorySystem& system,
    DeviceId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    if (m_fill_value.length == 0) {
        return DeviceEvent::null();
    }

    size_t element_size = m_fill_value.length;
    size_t count = m_layout.size_in_bytes / element_size;

    FillDescription description(m_fill_value);
    description.add_dimension(
        static_cast<memops_extent_type>(count),
        static_cast<memops_stride_type>(element_size)
    );

    return system.fill_device( //
        memory_id,
        reinterpret_cast<g_device_ptr_t>(m_ptr),
        description,
        stream_hint,
        deps
    );
}

}  // namespace kmm
