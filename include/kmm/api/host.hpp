#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "kmm/api/context.hpp"
#include "kmm/api/resource_guard.hpp"
#include "kmm/runtime/identifiers.hpp"

namespace kmm {

template<typename... Args>
class HostGuard;

class Host: public Context {
  public:
    Host(Runtime runtime, MemoryTransaction transaction = {}) :
        Context(std::move(runtime), std::move(transaction)) {}

    /// Acquires `args` and calls `fun(views...)`, where `views...` are the resolved views for
    /// `args...`.
    template<typename F, typename... Args>
    decltype(auto) submit(F&& fun, Args&&... args) {
        return access(std::forward<Args>(args)...).submit(std::forward<F>(fun));
    }

    template<typename... Args>
    HostGuard<Args...> access(Args&&... args) {
        return HostGuard<Args...>(runtime(), transaction(), std::forward<Args>(args)...);
    }
};

template<typename... Args>
class HostGuard {
  public:
    HostGuard(Runtime& runtime, MemoryTransaction parent, Args... args) :
        m_guard(runtime, MemoryId::host(), std::nullopt, parent, args...) {
        runtime.synchronize(m_guard.dependencies());
    }

    ~HostGuard() {
        m_guard.release();
    }

    Host context() const {
        return Host(m_guard.transaction());
    }

    template<size_t I>
    decltype(auto) get() {
        return m_guard.template get<I>();
    }

    decltype(auto) get() {
        static_assert(sizeof...(Args) == 1, "argument index not specified");
        return m_guard.template get<0>();
    }

    template<typename F>
    void submit(F fun) {
        m_guard.apply(std::move(fun));
    }

    template<typename F>
    void operator>>(F fun) {
        submit(std::move(fun));
    }

  private:
    ResourceGuard<Args...> m_guard;
};

}  // namespace kmm
