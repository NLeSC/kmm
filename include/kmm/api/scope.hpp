#pragma once

#include <exception>
#include <memory>
#include <tuple>
#include <utility>

#include "kmm/api/launch_arg.hpp"
#include "kmm/runtime/requisition.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

template<typename... Args>
class ScopeImpl {
  public:
    ScopeImpl(
        Runtime& runtime,
        MemoryId memory_id,
        std::optional<CUDAStreamRef> stream,
        MemoryTransaction parent,
        Args... args
    ) :
        m_args(args...),
        m_runtime(runtime) {
        acquire_impl(std::index_sequence_for<Args...> {}, runtime, memory_id);
        m_runtime.submit(m_req, stream, parent);
    }

    MemoryTransaction transaction() const noexcept {
        return m_req.transaction();
    }

    const DeviceEventSet& dependencies() const noexcept {
        return m_req.dependencies();
    }

    Runtime& runtime() noexcept {
        return m_runtime;
    }

    template<size_t I>
    decltype(auto) get() {
        return std::get<I>(m_args).resolve(m_runtime, m_req);
    }

    template<typename F, typename... Extra>
    decltype(auto) apply(F&& fun, Extra&&... extra) {
        return apply_impl(std::index_sequence_for<Args...> {}, fun, std::forward<Extra>(extra)...);
    }

    void release(DeviceEventSet deps = {}) {
        release_impl(std::index_sequence_for<Args...> {}, std::move(deps));
    }

    void poison(std::exception_ptr reason) {
        m_req.poison(m_runtime, std::move(reason));
    }

  private:
    template<size_t... Is>
    void acquire_impl(std::index_sequence<Is...>, Runtime& runtime, MemoryId memory_id) {
        (std::get<Is>(m_args).acquire(m_runtime, m_req, memory_id), ...);
    }

    template<size_t... Is, typename F, typename... Extra>
    decltype(auto) apply_impl(std::index_sequence<Is...>, F&& fun, Extra&&... extra) {
        return fun(
            std::forward<Extra>(extra)...,
            std::get<Is>(m_args).resolve(m_runtime, m_req)...
        );
    }

    template<size_t... Is>
    void release_impl(std::index_sequence<Is...>, DeviceEventSet deps) {
        (std::get<Is>(m_args).release(m_runtime, m_req), ...);
        m_runtime.release(m_req, std::move(deps));
    }

    std::tuple<LaunchArg<Args>...> m_args;
    Runtime m_runtime;
    Requisition m_req;
};

template<typename ContextT>
struct scope_invoke;

template<typename ContextT, typename... Args>
class Scope {
  public:
    Scope() = default;

    Scope(std::unique_ptr<ScopeImpl<Args...>> scope, std::unique_ptr<ContextT> context) :
        m_scope(std::move(scope)),
        m_context(std::move(context)) {}

    ~Scope() {
        if (m_scope) {
            release();
        }
    }

    void release() {
        m_scope->release(m_deps);
        m_scope = nullptr;
    }

    MemoryTransaction transaction() const noexcept {
        KMM_ASSERT(m_scope);
        return m_scope->transaction();
    }

    const DeviceEventSet& dependencies() const noexcept {
        KMM_ASSERT(m_scope);
        return m_scope->dependencies();
    }

    Runtime& runtime() noexcept {
        KMM_ASSERT(m_scope);
        return m_scope->runtime();
    }

    ContextT& context() noexcept {
        KMM_ASSERT(m_context);
        return *m_context;
    }

    template<size_t I>
    decltype(auto) get() {
        KMM_ASSERT(m_scope);
        return m_scope->template get<I>();
    }

    template<typename F>
    void apply(F&& fun) {
        KMM_ASSERT(m_scope);

        try {
            scope_invoke<ContextT> {}(*m_context, *m_scope, m_deps, std::forward<F>(fun));
        } catch (...) {
            m_scope->poison(std::current_exception());
            throw;
        }
    }

    template<typename F>
    void operator>>(F&& fun) {
        apply(std::forward<F>(fun));
    }

  protected:
    std::unique_ptr<ScopeImpl<Args...>> m_scope = nullptr;
    std::unique_ptr<ContextT> m_context;
    DeviceEventSet m_deps;
};

}  // namespace kmm
