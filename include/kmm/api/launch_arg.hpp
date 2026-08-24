#pragma once

#include <utility>
#include <vector>

#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/memory_manager.hpp"

namespace kmm {

/// Tags `value` as needing read-write access when passed to `Context::scope` (the default for a
/// plain argument is read-only access).
template<typename T>
struct Write {
    T& value;
};

template<typename T>
Write<T> write(T& value) {
    return Write<T> {value};
}

/// Tags `value` as needing read-only access when passed to `Context::scope`. A plain argument is
/// already read-only by default; this is only useful to force a `Write`-tagged value back down to
/// read-only.
template<typename T>
struct Read {
    T& value;
};

template<typename T>
Read<T> read(T& value) {
    return Read<T> {value};
}

template<typename T>
Read<T> read(Write<T> value) {
    return Read<T> {value.value};
}

template<typename T>
class Reduce;

class Runtime;
class Requisition;

/// Default `LaunchArg` implementation, used for any `T` without a more specific specialization
/// (e.g. a plain scalar or POD struct). The caller's value is copied once into this object; there
/// is no backing buffer to acquire or release. `LaunchArg<const T&>` and `LaunchArg<T&>` build on
/// top of this copy to expose it to the kernel by const or mutable reference instead of by value.
template<typename T>
class LaunchArg {
  public:
    using resolve_type = T;

    explicit LaunchArg(T value) : m_value(std::move(value)) {}

    void acquire(Runtime& runtime, Requisition& req, MemoryId memory_id) {}

    resolve_type resolve(Runtime& runtime, Requisition& req) {
        return m_value;
    }

    void release(Runtime& runtime, Requisition& req) {}

  protected:
    T m_value;
};

/// `LaunchArg` specialization used when `Context::scope` is given a const lvalue reference to
/// `T`. Just forwards to `LaunchArg<T>`.
template<typename T>
class LaunchArg<const T&>: public LaunchArg<T> {
  public:
    explicit LaunchArg(const T& value) : LaunchArg<T>(value) {}
};

/// See `LaunchArg<const T&>`.
template<typename T>
class LaunchArg<T&>: public LaunchArg<const T&> {
  public:
    explicit LaunchArg(T& value) : LaunchArg<const T&>(value) {}
};

}  // namespace kmm
