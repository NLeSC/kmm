#include <algorithm>
#include <utility>

#include "kmm/core/integer_fun.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interface.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/utils/gpu_utils.hpp"

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

size_t FlatDataInterface::size_in_bytes() const {
    return m_layout.size_in_bytes;
}

AllocResult FlatDataInterface::allocate(
    MemoryId memory_id,
    DeviceStream stream_hint,
    DeviceEventSet& deps_out
) {
    if (memory_id.is_host()) {
        KMM_ASSERT(m_host_ptr == nullptr);

        void* ptr = nullptr;
        auto result = m_system->allocate_host(m_layout.size_in_bytes, DeviceId(0), &ptr, deps_out);

        if (result == AllocResult::Success) {
            m_host_ptr = ptr;
        }

        return result;
    } else {
        auto id = memory_id.as_device();
        auto& ptr = m_device_ptrs[id.get()];
        KMM_ASSERT(ptr == 0);

        return m_system->allocate_device(id, m_layout.size_in_bytes, &ptr, deps_out);
    }
}

void FlatDataInterface::deallocate(
    MemoryId memory_id,
    DeviceStream stream_hint,
    const DeviceEventSet& deps
) {
    if (memory_id.is_host()) {
        KMM_ASSERT(m_host_ptr != nullptr);
        m_system->deallocate_host(m_host_ptr, m_layout.size_in_bytes, deps);
        m_host_ptr = nullptr;
    } else {
        auto id = memory_id.as_device();
        auto& ptr = m_device_ptrs[id.get()];
        KMM_ASSERT(ptr != 0);

        m_system->deallocate_device(id, ptr, m_layout.size_in_bytes, deps);
        ptr = 0;
    }
}

void* FlatDataInterface::address(MemoryId memory_id) const {
    if (memory_id.is_host()) {
        return m_host_ptr;
    } else {
        return (void*)m_device_ptrs[memory_id.as_device().get()];
    }
}

DeviceEvent FlatDataInterface::copy(
    MemoryId src,
    MemoryId dst,
    DeviceStream stream_hint,
    const DeviceEventSet& deps
) {
    size_t nbytes = m_layout.size_in_bytes;

    if (src.is_host() && dst.is_device()) {
        auto id = dst.as_device();
        return m_system->copy_host_to_device(  //
            id,
            m_host_ptr,
            m_device_ptrs[id.get()],
            nbytes,
            std::move(deps)
        );
    }

    if (src.is_device() && dst.is_host()) {
        auto id = src.as_device();
        return m_system->copy_device_to_host(  //
            id,
            m_device_ptrs[id.get()],
            m_host_ptr,
            nbytes,
            std::move(deps)
        );
    }

    if (src.is_device() && dst.is_device()) {
        auto src_id = src.as_device();
        auto dst_id = dst.as_device();

        return m_system->copy_device_to_device(
            src_id,
            dst_id,
            m_device_ptrs[src_id.get()],
            m_device_ptrs[dst_id.get()],
            nbytes,
            std::move(deps)
        );
    }

    KMM_PANIC("cannot copy from host memory to host memory");
}

bool FlatDataInterface::is_copy_supported(MemoryId src, MemoryId dst) {
    return m_system->is_copy_supported(src, dst);
}

std::future<void> FlatDataInterface::initialize_host(const DeviceEventSet& deps) {
    if (m_fill_value.length != 0) {
        for (const auto& event : deps) {
            event.synchronize();
        }

        size_t element_size = m_fill_value.length;
        size_t count = m_layout.size_in_bytes / element_size;

        FillDescription description;
        description.value = m_fill_value;
        description.add_dimension(
            static_cast<memops_extent_type>(count),
            static_cast<memops_stride_type>(element_size)
        );

        fill(m_host_ptr, description);
    }

    return {};
}

DeviceEvent FlatDataInterface::initialize_device(
    DeviceId memory_id,
    DeviceStream stream_hint,
    const DeviceEventSet& deps
) {
    if (m_fill_value.length == 0) {
        return DeviceEvent::null();
    }

    auto stream = m_system->stream(memory_id);
    void* dst_addr = (void*)m_device_ptrs[memory_id.get()];

    size_t element_size = m_fill_value.length;
    size_t count = m_layout.size_in_bytes / element_size;

    CUDAContextGuard guard {stream.context()};

    return stream.with_stream(deps, [&](CUstream cuda_stream) {
        FillDescription description;
        description.value = m_fill_value;
        description.add_dimension(
            static_cast<memops_extent_type>(count),
            static_cast<memops_stride_type>(element_size)
        );

        fill_async(cuda_stream, dst_addr, description);
    });
}

}  // namespace kmm
