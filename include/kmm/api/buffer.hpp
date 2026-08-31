#pragma once

#include <exception>
#include <string>

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memops/copy.hpp"
#include "kmm/runtime/runtime.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

class Runtime;

class Buffer {
  public:
    struct Impl;

    Buffer() = default;

    /// Create a new runtime-managed buffer. The runtime owns the allocation and is free to place,
    /// move, and evict it across memories as needed.
    static Buffer create(
        Runtime runtime,
        BufferLayout layout,
        std::string name,
        FillValue fill_value = {},
        std::optional<MemoryId> home = {},
        std::optional<BufferKind> kind = std::nullopt
    );

    /// Wrap a pre-existing, externally-owned allocation as a buffer, without copying. KMM never
    /// allocates, frees, or relocates the memory; the caller must keep `ptr` alive for as long as
    /// the returned `Buffer` (or any copy of it) is in use. See `Runtime::adopt_buffer`.
    static Buffer adopt(
        Runtime runtime,
        BufferLayout layout,
        void* ptr,
        MemoryId memory_id,
        std::string name = {}
    );

    BufferId id() const;
    BufferLayout layout() const;
    Runtime runtime() const;

    /// Returns this buffer's home memory, if it has one. See `Runtime::buffer_home`.
    std::optional<MemoryId> home() const;

    void prefetch(MemoryId memory_id, bool invalidate_others = false) const;
    void poison(std::exception_ptr reason) const;
    void invalidate() const;

    void copy_to(
        void* dest,
        size_t nbytes,
        size_t offset = 0,
        MemoryId memory_id = MemoryId::host()
    ) const;

    void copy_from(
        const void* dest,
        size_t nbytes,
        size_t offset = 0,
        MemoryId memory_id = MemoryId::host()
    ) const;

    void copy_to(  //
        void* dest,
        CopyDescription description,
        MemoryId memory_id = MemoryId::host()
    ) const;

    void copy_from(
        const void* src,
        CopyDescription description,
        MemoryId memory_id = MemoryId::host()
    ) const;

    explicit operator bool() const {
        return bool(m_impl);
    }

  private:
    refcnt_ptr<Impl> m_impl;
};

KMM_REFCNT_TRAITS_FWD(Buffer::Impl)

}  // namespace kmm
