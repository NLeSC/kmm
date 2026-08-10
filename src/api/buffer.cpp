#include "kmm/api/buffer.hpp"
#include "kmm/api/context.hpp"

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

Buffer::Buffer(Runtime runtime, BufferLayout layout, std::string name, FillValue fill_value) {
    auto id = runtime.create_buffer(layout, std::move(name), fill_value);
    m_impl = make_refcnt<Impl>(runtime, id, layout);
}

BufferId Buffer::id() const {
    KMM_ASSERT(m_impl);
    return m_impl->id;
}

Runtime Buffer::runtime() const {
    KMM_ASSERT(m_impl);
    return m_impl->runtime;
}

BufferLayout Buffer::layout() const {
    KMM_ASSERT(m_impl);
    return m_impl->layout;
}

void Buffer::prefetch(MemoryId memory_id, AccessMode mode) const {
    if (m_impl) {
        m_impl->runtime.prefetch_buffer(m_impl->id, memory_id, mode);
    }
}

void Buffer::poison(std::exception_ptr reason) const {
    KMM_ASSERT(m_impl);
    m_impl->runtime.poison_buffer(m_impl->id, std::move(reason));
}

void Buffer::invalidate() const {
    KMM_ASSERT(m_impl);
    m_impl->runtime.invalidate_buffer(id());
}

void Buffer::copy_to(void* dest, size_t nbytes, size_t offset) const {
    KMM_ASSERT(m_impl);
    auto runtime = m_impl->runtime;

    Requisition req {MemoryId::host()};
    req.add(id(), AccessMode::Read);
    runtime.submit(req);

    auto accessor = req.accessor(runtime, 0);
    KMM_ASSERT(accessor.memory_id == MemoryId::host());
    KMM_ASSERT(nbytes <= accessor.size_in_bytes);
    KMM_ASSERT(offset <= accessor.size_in_bytes - nbytes);

    memcpy(dest, static_cast<std::byte*>(accessor.address) + offset, nbytes);

    runtime.release(req);
}

void Buffer::copy_from(const void* dest, size_t nbytes, size_t offset) const {
    auto runtime = m_impl->runtime;

    Requisition req {MemoryId::host()};
    req.add(id(), AccessMode::ReadWrite);
    runtime.submit(req);

    auto accessor = req.accessor(runtime, 0);
    KMM_ASSERT(accessor.memory_id == MemoryId::host());
    KMM_ASSERT(nbytes <= accessor.size_in_bytes);
    KMM_ASSERT(offset <= accessor.size_in_bytes - nbytes);
    KMM_ASSERT(accessor.is_writable);

    memcpy(static_cast<std::byte*>(accessor.address) + offset, dest, nbytes);

    runtime.release(req);
}

KMM_REFCNT_TRAITS_IMPL(Buffer::Impl)

}  // namespace kmm
