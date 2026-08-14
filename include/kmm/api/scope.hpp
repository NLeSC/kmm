#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kmm/api/launch_arg.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/runtime/requisition.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

template<typename... Args>
struct ScopeImpl {
    using args_tuple = std::tuple<LaunchArg<std::decay_t<Args>>...>;

    template<typename ParentCtx>
    explicit ScopeImpl(
        ParentCtx& ctx,
        MemoryId memory_id,
        MemoryTransaction transaction,
        Args&&... args
    ) :
        req(memory_id, std::move(transaction)),
        args(LaunchArg<std::decay_t<Args>>(std::forward<Args>(args))...),
        runtime(ctx.runtime()) {
        acquire_and_submit(ctx);
    }

    ~ScopeImpl() {
        std::apply([&](auto&... a) { (a.release(runtime, req), ...); }, args);
        runtime.release(req);
    }

    template<typename ParentCtx>
    void acquire_and_submit(ParentCtx& ctx) {
        std::apply([&](auto&... a) { (a.acquire(ctx.runtime(), req), ...); }, args);
        ctx.runtime().submit(ctx.submit_stream(), req);
    }

    Requisition req;
    args_tuple args;
    Runtime runtime;
};

template<typename Ctx, typename... Args>
class Scope {
    KMM_NOT_COPYABLE(Scope)

  public:
    template<typename ParentCtx>
    explicit Scope(
        ParentCtx& ctx,
        MemoryId memory_id,
        MemoryTransaction transaction,
        Args&&... args
    ) :
        m_state(std::make_unique<ScopeImpl<Args...>>(
            ctx,
            memory_id,
            std::move(transaction),
            std::forward<Args>(args)...
        )),
        m_ctx(ctx.child(m_state->req.transaction())) {}

    void finish() {
        m_state.reset();
    }

    void poison(std::exception_ptr reason) {
        m_state->req.poison(m_state->runtime, std::move(reason));
    }

    template<size_t I>
    decltype(auto) get() {
        return std::get<I>(m_state->args).resolve(m_state->runtime, m_state->req);
    }

    template<typename Launcher>
    decltype(auto) apply(Launcher&& launcher) {
        try {
            return apply_impl(
                std::forward<Launcher>(launcher),
                std::make_index_sequence<sizeof...(Args)>()
            );
        } catch (...) {
            poison(std::current_exception());
            throw;
        }
    }

  private:
    template<typename Launcher, size_t... Is>
    decltype(auto) apply_impl(Launcher&& launcher, std::index_sequence<Is...>) {
        auto guard = m_ctx.activate();
        return launcher(
            m_ctx,
            std::get<Is>(m_state->args).resolve(m_state->runtime, m_state->req)...
        );
    }

    std::unique_ptr<ScopeImpl<Args...>> m_state;
    Ctx m_ctx;
};

template<typename Ctx, typename Arg>
class Scope<Ctx, Arg> {
    KMM_NOT_COPYABLE(Scope)

    using view_type =
        typename std::tuple_element_t<0, typename ScopeImpl<Arg>::args_tuple>::resolve_type;

  public:
    template<typename ParentCtx>
    explicit Scope(ParentCtx& ctx, MemoryId memory_id, MemoryTransaction transaction, Arg&& arg) :
        m_state(std::make_unique<ScopeImpl<Arg>>(
            ctx,
            memory_id,
            std::move(transaction),
            std::forward<Arg>(arg)
        )),
        m_ctx(ctx.child(m_state->req.transaction())) {}

    void finish() {
        m_state.reset();
    }

    void poison(std::exception_ptr reason) {
        m_state->req.poison(m_state->runtime, std::move(reason));
    }

    template<size_t I = 0>
    view_type get() {
        static_assert(I == 0, "Scope with a single argument only has index 0");
        return std::get<I>(m_state->args).resolve(m_state->runtime, m_state->req);
    }

    operator view_type() {
        return get();
    }

    template<typename Launcher>
    decltype(auto) apply(Launcher&& launcher) {
        try {
            auto guard = m_ctx.activate();
            return launcher(
                m_ctx,
                std::get<0>(m_state->args).resolve(m_state->runtime, m_state->req)
            );
        } catch (...) {
            poison(std::current_exception());
            throw;
        }
    }

  private:
    std::unique_ptr<ScopeImpl<Arg>> m_state;
    Ctx m_ctx;
};

/// Shorthand for `scope.apply(fun)`.
template<typename Ctx, typename... Args, typename Launcher>
decltype(auto) operator>>(Scope<Ctx, Args...>& scope, Launcher&& launcher) {
    return scope.apply(std::forward<Launcher>(launcher));
}

/// Shorthand for `scope.apply(fun)`, for temporary scopes.
template<typename Ctx, typename... Args, typename Launcher>
decltype(auto) operator>>(Scope<Ctx, Args...>&& scope, Launcher&& launcher) {
    return scope.apply(std::forward<Launcher>(launcher));
}

}  // namespace kmm
