#pragma once

#include <exception>
#include <string>

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/runtime.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

class Runtime;

class Buffer {
  public:
    struct Impl;

    Buffer() = default;

    /// Allocates a new buffer of `layout` within `context`. The buffer is released (via
    /// `context`) once the last reference to it is dropped. If `fill_value` is non-empty, the
    /// buffer's contents are set to repeated copies of it the first time it is materialized in
    /// any memory.
    Buffer(
        Runtime runtime,
        BufferLayout layout,
        std::string name,
        FillValue fill_value = {},
        std::optional<MemoryId> home = {}
    );

    BufferId id() const;
    BufferLayout layout() const;
    Runtime runtime() const;
    void prefetch(MemoryId memory_id, bool invalidate_others = false) const;
    void poison(std::exception_ptr reason) const;
    void invalidate() const;
    void copy_to(void* dest, size_t nbytes, size_t offset = 0) const;
    void copy_from(const void* dest, size_t nbytes, size_t offset = 0) const;

    explicit operator bool() const {
        return bool(m_impl);
    }

  private:
    refcnt_ptr<Impl> m_impl;
};

KMM_REFCNT_TRAITS_FWD(Buffer::Impl)

}  // namespace kmm
