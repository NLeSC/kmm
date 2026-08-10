#pragma once

#include <string>
#include <utility>
#include <vector>

#include "kmm/api/array.hpp"
#include "kmm/api/buffer.hpp"
#include "kmm/api/reduce.hpp"
#include "kmm/api/scope.hpp"
#include "kmm/core/checked_compare.hpp"
#include "kmm/core/layout.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

/// The handle through which arrays are allocated and accessed.
class Context {
  public:
    /// Implicit conversion from `Runtime`, so `Runtime` can be passed where `Context` is expected.
    Context(Runtime runtime) : m_runtime(std::move(runtime)) {}

    /// Allocates a new, unpopulated buffer of `layout`. The returned `Buffer` releases itself
    /// (via this context) once its last reference is dropped.
    Buffer allocate_buffer(BufferLayout layout, std::string name) {
        return Buffer(runtime(), std::move(layout), std::move(name));
    }

    /// Allocates a new, uninitialized array of the given shape, without setting its content.
    template<typename T, typename PolicyT = RowMajor, typename... Sizes>
    Array<T, sizeof...(Sizes), PolicyT> empty(Sizes... extents) {
        return DomainArray<T, Shape<sizeof...(Sizes)>, PolicyT>(
            runtime(),
            Shape<sizeof...(Sizes)> {checked_cast<default_index_type>(extents)...},
            PolicyT {}
        );
    }

    /// Allocates a new array of the given shape fill with the given value.
    template<typename T, typename PolicyT = RowMajor, typename... Sizes>
    Array<T, sizeof...(Sizes), PolicyT> fill(T value, Sizes... extents) {
        return DomainArray<T, Shape<sizeof...(Sizes)>, PolicyT>(
            runtime(),
            Shape<sizeof...(Sizes)> {checked_cast<default_index_type>(extents)...},
            PolicyT {},
            value
        );
    }

    /// Allocates a new array of the given shape with zeros.
    template<typename T, typename PolicyT = RowMajor, typename... Sizes>
    Array<T, sizeof...(Sizes), PolicyT> zeros(Sizes... extents) {
        return fill<T, PolicyT>(T {0}, extents...);
    }

    /// Allocates a new array of the given shape filled with ones.
    template<typename T, typename PolicyT = RowMajor, typename... Sizes>
    Array<T, sizeof...(Sizes), PolicyT> ones(Sizes... extents) {
        return fill<T, PolicyT>(T {1}, extents...);
    }

    /// Allocates a new 1-D array and copies the contents of `values` into it.
    template<typename T, typename PolicyT = RowMajor>
    Array<T, 1, PolicyT> from_vector(const std::vector<T>& values) {
        auto array = empty<T, PolicyT>(values.size());
        array.buffer().copy_from(values.data(), values.size() * sizeof(T));
        return array;
    }

    /// The `Runtime` backing this context.
    Runtime& runtime() noexcept {
        return m_runtime;
    }

    /// Equivalent to `scope(MemoryId::host(), callback, args...)`.
    template<typename F, typename... Args>
    decltype(auto) scope(F&& callback, Args&&... args) {
        return scope(preferred_memory_id(), std::forward<F>(callback), std::forward<Args>(args)...);
    }

    template<typename F, typename... Args>
    decltype(auto) scope(MemoryId memory_id, F&& callback, Args&&... args) {
        Scope<Args...> scope {
            runtime(),
            memory_id,
            m_root_transaction,
            std::forward<Args>(args)...};

        return std::apply(std::forward<F>(callback), scope.resolve());
    }

    virtual MemoryId preferred_memory_id() const {
        return MemoryId::host();
    }

  private:
    MemoryTransaction m_root_transaction;
    Runtime m_runtime;
};

}  // namespace kmm

#include "kmm/api/scope.hpp"
