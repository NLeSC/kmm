#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/resource.hpp"

namespace kmm {

/// Tags `value` as needing read-write access when passed to `Device::scope`/`Host::scope` (the
/// default for a plain argument is read-only access).
template<typename T>
struct Write {
    T& value;
};

template<typename T>
Write<T> write(T& value) {
    return Write<T> {value};
}

/// Tags `value` as needing read-only access when passed to `Device::scope`/`Host::scope`. A plain
/// argument is already read-only by default; this is only useful to force a `Write`-tagged value
/// back down to read-only.
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

class Runtime;

/// Default `LaunchArg` implementation, used for any `T` without a more specific specialization
/// (e.g. a plain scalar or POD struct). The caller's value is copied once into this object; there
/// is no backing buffer to acquire or release. `LaunchArg<const T&>` and `LaunchArg<T&>` build on
/// top of this copy to expose it to the kernel by const or mutable reference instead of by value.
template<typename T>
class LaunchArg {
  public:
    using resolve_type = T;

    explicit LaunchArg(T value) : m_value(std::move(value)) {}

    void acquire(Runtime& runtime, ResourceRequest& requests, MemoryId memory_id) {}

    resolve_type resolve(Runtime& runtime, const ResourceGrant& grant) {
        return m_value;
    }

    /// Called once after the launch's resource grant has been released (see
    /// `ResourceGuard::release`). At this point the buffers touched by the launch are free again,
    /// so an override may submit follow-up work on them; the default is to do nothing.
    void release(Runtime& runtime) {}

  protected:
    T m_value;
};

/// `LaunchArg` specialization used when `Device::scope`/`Host::scope` is given a const lvalue
/// reference to `T`. Just forwards to `LaunchArg<T>`.
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

/// `LaunchArg` specialization for `std::tuple<Ts...>`. Each element `Ts` is wrapped in its own
/// `LaunchArg<Ts>` (so, for example, a tuple of `Write<Array<T>>` elements acquires/releases each
/// array independently), and `resolve()` returns a `std::tuple` of the per-element resolved
/// values, in the same order.
template<typename... Ts>
class LaunchArg<std::tuple<Ts...>> {
  public:
    using resolve_type = std::tuple<typename LaunchArg<Ts>::resolve_type...>;

    explicit LaunchArg(std::tuple<Ts...> value) : m_args(std::move(value)) {}

    void acquire(Runtime& runtime, ResourceRequest& requests, MemoryId memory_id) {
        acquire_impl(std::index_sequence_for<Ts...> {}, runtime, requests, memory_id);
    }

    resolve_type resolve(Runtime& runtime, const ResourceGrant& grant) {
        return resolve_impl(std::index_sequence_for<Ts...> {}, runtime, grant);
    }

    void release(Runtime& runtime) {
        release_impl(std::index_sequence_for<Ts...> {}, runtime);
    }

  private:
    template<size_t... Is>
    void acquire_impl(
        std::index_sequence<Is...>,
        Runtime& runtime,
        ResourceRequest& requests,
        MemoryId memory_id
    ) {
        (std::get<Is>(m_args).acquire(runtime, requests, memory_id), ...);
    }

    template<size_t... Is>
    resolve_type resolve_impl(
        std::index_sequence<Is...>,
        Runtime& runtime,
        const ResourceGrant& grant
    ) {
        return resolve_type {std::get<Is>(m_args).resolve(runtime, grant)...};
    }

    template<size_t... Is>
    void release_impl(std::index_sequence<Is...>, Runtime& runtime) {
        (std::get<Is>(m_args).release(runtime), ...);
    }

    std::tuple<LaunchArg<Ts>...> m_args;
};

}  // namespace kmm
