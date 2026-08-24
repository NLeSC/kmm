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

void Buffer::copy_to(void* dest, size_t nbytes, size_t offset) const {
    assert_impl(m_impl);
    auto runtime = m_impl->runtime;

    Requisition req;
    req.add(MemoryId::host(), id(), AccessMode::Read);
    runtime.submit(req);

    auto accessor = req.accessor(0);
    KMM_ASSERT(accessor.memory_id == MemoryId::host());
    KMM_ASSERT(nbytes <= accessor.size_in_bytes);
    KMM_ASSERT(offset <= accessor.size_in_bytes - nbytes);

    memcpy(dest, static_cast<std::byte*>(accessor.address) + offset, nbytes);

    runtime.release(req);
}

void Buffer::copy_from(const void* dest, size_t nbytes, size_t offset) const {
    auto runtime = m_impl->runtime;

    Requisition req;
    req.add(MemoryId::host(), id(), AccessMode::ReadWrite);
    runtime.submit(req);

    auto accessor = req.accessor(0);
    KMM_ASSERT(accessor.memory_id == MemoryId::host());
    KMM_ASSERT(nbytes <= accessor.size_in_bytes);
    KMM_ASSERT(offset <= accessor.size_in_bytes - nbytes);
    KMM_ASSERT(accessor.is_writable);

    memcpy(static_cast<std::byte*>(accessor.address) + offset, dest, nbytes);

    runtime.release(req);
}

KMM_REFCNT_TRAITS_IMPL(Buffer::Impl)

}  // namespace kmm
