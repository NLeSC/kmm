#include <cstddef>

#include "spdlog/spdlog.h"

#include "kmm/api/buffer.hpp"
#include "kmm/api/context.hpp"
#include "kmm/runtime/resource.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

struct Buffer::Impl: reference_count<Impl> {
    Impl(Runtime rt, BufferId id, BufferLayout layout) :
        runtime(std::move(rt)),
        id(id),
        layout(layout) {}

    ~Impl() {
        runtime.release_buffer(id);
    }

    Runtime runtime;
    BufferId id;
    BufferLayout layout;
};

Buffer::Buffer(
    Runtime runtime,
    BufferLayout layout,
    std::string name,
    FillValue fill_value,
    std::optional<MemoryId> home
) {
    auto id = runtime.create_buffer(layout, std::move(name), fill_value, home);
    m_impl = make_refcnt<Impl>(runtime, id, layout);
}

static void assert_impl(const refcnt_ptr<Buffer::Impl>& impl) {
    if (impl == nullptr) {
        throw std::runtime_error(
            "cannot access buffer as it has not yet been registered with a runtime system"
        );
    }
}

BufferId Buffer::id() const {
    assert_impl(m_impl);
    return m_impl->id;
}

Runtime Buffer::runtime() const {
    assert_impl(m_impl);
    return m_impl->runtime;
}

BufferLayout Buffer::layout() const {
    assert_impl(m_impl);
    return m_impl->layout;
}

std::optional<MemoryId> Buffer::home() const {
    assert_impl(m_impl);
    return m_impl->runtime.buffer_home(m_impl->id);
}

void Buffer::prefetch(MemoryId memory_id, bool invalidate_others) const {
    if (m_impl) {
        m_impl->runtime.prefetch_buffer(m_impl->id, memory_id, invalidate_others);
    }
}

void Buffer::poison(std::exception_ptr reason) const {
    assert_impl(m_impl);
    m_impl->runtime.poison_buffer(m_impl->id, std::move(reason));
}

void Buffer::invalidate() const {
    assert_impl(m_impl);
    m_impl->runtime.invalidate_buffer(id());
}

static void copy_device_to_host(
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& simplified
) {
    KMM_ASSERT(simplified.num_dims == 0);

    auto src = static_cast<const std::byte*>(src_addr) + simplified.src_offset;
    auto dst = static_cast<std::byte*>(dst_addr) + simplified.dst_offset;

    KMM_GPU_CHECK(gpuMemcpyDtoH(dst, (GPUDeviceptr)src, simplified.element_size));
}

static void copy_host_to_device(
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& simplified
) {
    KMM_ASSERT(simplified.num_dims == 0);

    auto src = static_cast<const std::byte*>(src_addr) + simplified.src_offset;
    auto dst = static_cast<std::byte*>(dst_addr) + simplified.dst_offset;

    KMM_GPU_CHECK(gpuMemcpyHtoD((GPUDeviceptr)dst, src, simplified.element_size));
}

static CopyDescription contiguous_copy_description(
    size_t nbytes,
    size_t src_offset,
    size_t dst_offset
) {
    CopyDescription description;
    description.src_offset = checked_cast<memops_stride_type>(src_offset);
    description.dst_offset = checked_cast<memops_stride_type>(dst_offset);
    description.add_dimension(checked_cast<memops_extent_type>(nbytes), 1, 1);
    return description;
}

void Buffer::copy_to(void* dest, size_t nbytes, size_t offset, MemoryId memory_id) const {
    copy_to(dest, contiguous_copy_description(nbytes, offset, 0), memory_id);
}

void Buffer::copy_to(void* dest, CopyDescription description, MemoryId memory_id) const {
    assert_impl(m_impl);
    auto runtime = m_impl->runtime;

    auto simplified = description.simplify();
    if (!memory_id.is_host() && simplified.num_dims != 0) {
        spdlog::warn("copy could not be reduced to a 1D copy, falling back to a copy via host");
        memory_id = MemoryId::host();
    }

    ResourceRequest requests;
    requests.add(memory_id, id(), AccessMode::Read);
    auto grant = runtime.submit(std::move(requests));

    auto accessor = grant.accessor(0);
    KMM_ASSERT(accessor.memory_id == memory_id);

    auto src_range = description.src_range();
    KMM_ASSERT(
        src_range.start >= 0 && static_cast<size_t>(src_range.stop) <= accessor.size_in_bytes
    );

    if (memory_id.is_host()) {
        copy(accessor.address, dest, description);
    } else {
        copy_device_to_host(accessor.address, dest, simplified);
    }

    runtime.release(grant);
}

void Buffer::copy_from(const void* dest, size_t nbytes, size_t offset, MemoryId memory_id) const {
    copy_from(dest, contiguous_copy_description(nbytes, 0, offset), memory_id);
}

void Buffer::copy_from(const void* src, CopyDescription description, MemoryId memory_id) const {
    assert_impl(m_impl);
    auto runtime = m_impl->runtime;

    auto simplified = description.simplify();
    if (!memory_id.is_host() && simplified.num_dims != 0) {
        spdlog::warn("copy could not be reduced to a 1D copy, falling back to a copy via host");
        memory_id = MemoryId::host();
    }

    ResourceRequest requests;
    requests.add(memory_id, id(), AccessMode::ReadWrite);
    auto grant = runtime.submit(std::move(requests));

    auto accessor = grant.accessor(0);
    KMM_ASSERT(accessor.memory_id == memory_id);
    KMM_ASSERT(accessor.is_writable);

    auto dst_range = description.dst_range();
    KMM_ASSERT(
        dst_range.start >= 0 && static_cast<size_t>(dst_range.stop) <= accessor.size_in_bytes
    );

    if (memory_id.is_host()) {
        copy(src, accessor.address, description);
    } else {
        copy_host_to_device(src, accessor.address, simplified);
    }

    runtime.release(grant);
}

KMM_REFCNT_TRAITS_IMPL(Buffer::Impl)

}  // namespace kmm
