#include <algorithm>
#include <utility>

#include "kmm/core/integer_fun.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interfaces/flat.hpp"
#include "kmm/runtime/memops/fill.hpp"

namespace kmm {

static BufferLayout normalize_buffer_layout(BufferLayout layout) {
    static constexpr size_t max_align = 128;
    size_t k = std::min(std::max(layout.size_in_bytes, layout.alignment), max_align);
    size_t align = round_up_to_power_of_two(k);
    return {round_up_to_multiple(layout.size_in_bytes, align), align};
}

FlatDataInterface::FlatDataInterface(
    BufferLayout layout,
    refcnt_ptr<MemorySystem> system,
    FillValue fill_value
) :
    m_layout(normalize_buffer_layout(layout)),
    m_system(std::move(system)),
    m_fill_value(std::move(fill_value)) {}

size_t FlatDataInterface::size_in_bytes() const noexcept {
    return m_layout.size_in_bytes;
}

AllocResult FlatDataInterface::allocate(
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    DeviceEventSet& deps_out
) {
    if (memory_id.is_host()) {
        KMM_ASSERT(m_host_ptr == nullptr);
        return m_system->allocate_host(m_layout, &m_host_ptr, stream_hint, deps_out);
    } else {
        auto id = memory_id.as_device();
        auto& ptr = m_device_ptrs[id.get()];
        KMM_ASSERT(ptr == 0);
        return m_system->allocate_device(id, m_layout, &ptr, stream_hint, deps_out);
    }
}

void FlatDataInterface::deallocate(
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    if (memory_id.is_host()) {
        KMM_ASSERT(m_host_ptr != nullptr);
        m_system->deallocate_host(m_host_ptr, m_layout, stream_hint, deps);
        m_host_ptr = nullptr;
    } else {
        auto id = memory_id.as_device();
        auto& ptr = m_device_ptrs[id.get()];
        KMM_ASSERT(ptr != 0);

        m_system->deallocate_device(id, ptr, m_layout, stream_hint, deps);
        ptr = 0;
    }
}

void* FlatDataInterface::address(MemoryId memory_id) const noexcept {
    if (memory_id.is_host()) {
        return m_host_ptr;
    } else {
        return (void*)m_device_ptrs[memory_id.as_device().get()];
    }
}

void FlatDataInterface::copy(
    MemoryId src,
    MemoryId dst,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps,
    DeviceEventSet& deps_out
) {
    size_t nbytes = m_layout.size_in_bytes;

    if (src.is_host() && dst.is_device()) {
        auto id = dst.as_device();
        auto event = m_system->copy_host_to_device(  //
            id,
            m_host_ptr,
            m_device_ptrs[id.get()],
            nbytes,
            stream_hint,
            deps
        );

        deps_out.insert(event);
        return;
    }

    if (src.is_device() && dst.is_host()) {
        auto id = src.as_device();

        auto event = m_system->copy_device_to_host(  //
            id,
            m_device_ptrs[id.get()],
            m_host_ptr,
            nbytes,
            stream_hint,
            deps
        );

        deps_out.insert(event);
        return;
    }

    if (src.is_device() && dst.is_device()) {
        auto src_id = src.as_device();
        auto dst_id = dst.as_device();

        auto event = m_system->copy_device_to_device(
            src_id,
            dst_id,
            m_device_ptrs[src_id.get()],
            m_device_ptrs[dst_id.get()],
            nbytes,
            stream_hint,
            deps
        );

        deps_out.insert(event);
        return;
    }

    KMM_PANIC("cannot copy from host memory to host memory");
}

bool FlatDataInterface::is_copy_supported(MemoryId src, MemoryId dst) const noexcept {
    return m_system->is_copy_supported(src, dst);
}

std::future<void> FlatDataInterface::initialize_host(const DeviceEventSet& deps) {
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

    return m_system->fill_host(m_host_ptr, description, deps);
}

DeviceEvent FlatDataInterface::initialize_device(
    DeviceId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    if (m_fill_value.length == 0) {
        return DeviceEvent::null();
    }

    KMM_ASSERT(m_layout.size_in_bytes % m_fill_value.length);
    size_t element_size = m_fill_value.length;
    size_t count = m_layout.size_in_bytes / m_fill_value.length;

    FillDescription description(m_fill_value);
    description.add_dimension(
        static_cast<memops_extent_type>(count),
        static_cast<memops_stride_type>(element_size)
    );

    return m_system->fill_device(
        memory_id,
        m_device_ptrs[memory_id.get()],
        description,
        stream_hint,
        deps
    );
}

AllocResult FlatDataInterface::allocate_and_copy(
    MemoryId src,
    MemoryId dst,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps_in,
    DeviceEventSet& deps_out
) {
    if (src.is_host() && dst.is_device()) {
        auto id = dst.as_device();
        auto& ptr = m_device_ptrs[id.get()];
        KMM_ASSERT(ptr == 0);

        auto dep_out = DeviceEvent {};
        auto result = m_system->allocate_device_and_copy_from_host(  //
            id,
            m_layout,
            &m_device_ptrs[id.get()],
            m_host_ptr,
            stream_hint,
            deps_in,
            dep_out
        );

        deps_out.insert(dep_out);
        return result;
    }

    if (src.is_device() && dst.is_host()) {
        DeviceEventSet deps;
        auto id = src.as_device();
        KMM_ASSERT(m_host_ptr == nullptr);

        auto dep_out = DeviceEvent {};
        auto result = m_system->allocate_host_and_copy_from_device(
            m_layout,
            &m_host_ptr,
            id,
            m_device_ptrs[id.get()],
            stream_hint,
            deps_in,
            dep_out
        );

        deps_out.insert(dep_out);
        return result;
    }

    // just forward to the default impl.
    return DataInterface::allocate_and_copy(src, dst, stream_hint, deps_in, deps_out);
}

}  // namespace kmm
