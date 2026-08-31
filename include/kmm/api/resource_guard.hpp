#pragma once

#include <exception>
#include <memory>
#include <tuple>
#include <utility>

#include "kmm/api/launch_arg.hpp"
#include "kmm/runtime/resource.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

template<typename... Args>
class ResourceGuard {
  public:
    ResourceGuard(
        Runtime& runtime,
        MemoryId memory_id,
        std::optional<GPUStreamRef> stream,
        MemoryTransaction parent,
        Args... args
    ) {
        std::tuple<LaunchArg<Args>...> args_tuple(args...);

        // `ResourceGrant` is not movable, so it must be initialized in place.
        m_state = std::unique_ptr<State>(new State {
            runtime,
            submit_impl(
                std::index_sequence_for<Args...> {},
                args_tuple,
                runtime,
                memory_id,
                stream,
                std::move(parent)
            ),
            std::move(args_tuple)
        });
    }

    Runtime& runtime() const noexcept {
        return m_state->runtime;
    }

    const MemoryTransaction& transaction() const noexcept {
        return m_state->grant.transaction();
    }

    const DeviceEventSet& dependencies() const noexcept {
        return m_state->grant.dependencies();
    }

    template<size_t I>
    decltype(auto) get() {
        return std::get<I>(m_state->args).resolve(m_state->runtime, m_state->grant);
    }

    template<typename F, typename... Extra>
    decltype(auto) apply(F&& fun, Extra&&... extra) {
        try {
            return apply_impl(
                std::index_sequence_for<Args...> {},
                std::forward<F>(fun),
                std::forward<Extra>(extra)...
            );
        } catch (...) {
            poison(std::current_exception());
            throw;
        }
    }

    void release(DeviceEventSet deps = {}) {
        std::exception_ptr error;

        // The grant is released first so that each arg's `release()` runs as a genuine
        // post-completion hook: by the time it is called the buffers this launch touched are no
        // longer held by the grant, so `release()` may itself submit follow-up work on them
        // (e.g. `reduce(array, k)` folding its scratch buffer into the destination array).
        try {
            m_state->runtime.release(m_state->grant, std::move(deps));
        } catch (...) {
            error = std::current_exception();
        }

        // Every arg's `release()` must still run even when an earlier step threw, otherwise a
        // single failure would strand the buffers held by the remaining args. The first error
        // seen (from the grant release or any arg) is rethrown once every arg has had its turn.
        release_impl(std::index_sequence_for<Args...> {}, error);

        if (error) {
            std::rethrow_exception(error);
        }
    }

    void poison(std::exception_ptr reason) noexcept {
        m_state->runtime.poison(m_state->grant, reason);
    }

  private:
    struct State {
        Runtime runtime;
        ResourceGrant grant;
        std::tuple<LaunchArg<Args>...> args;
    };

    template<size_t... Is>
    static ResourceGrant submit_impl(
        std::index_sequence<Is...>,
        std::tuple<LaunchArg<Args>...>& args,
        Runtime& runtime,
        MemoryId memory_id,
        std::optional<GPUStreamRef> stream,
        MemoryTransaction parent
    ) {
        ResourceRequest requests;
        (std::get<Is>(args).acquire(runtime, requests, memory_id), ...);
        return runtime.submit(std::move(requests), stream, std::move(parent));
    }

    template<size_t... Is, typename F, typename... Extra>
    decltype(auto) apply_impl(std::index_sequence<Is...>, F&& fun, Extra&&... extra) {
        return fun(
            std::forward<Extra>(extra)...,
            std::get<Is>(m_state->args).resolve(m_state->runtime, m_state->grant)...
        );
    }

    template<size_t... Is>
    void release_impl(std::index_sequence<Is...>, std::exception_ptr& error) {
        (release_one(std::get<Is>(m_state->args), error), ...);
    }

    template<typename Arg>
    void release_one(Arg& arg, std::exception_ptr& error) {
        try {
            arg.release(m_state->runtime);
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
        }
    }

    std::unique_ptr<State> m_state;
};

}  // namespace kmm
