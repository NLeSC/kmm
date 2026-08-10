#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kmm/api/launch_arg.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/requisition.hpp"
#include "kmm/runtime/runtime.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

template<typename... Args>
class Scope {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scope)

  public:
    explicit Scope(Runtime& runtime, MemoryId memory_id, MemoryTransaction parent, Args&&... args) :
        m_runtime(runtime),
        m_requisition(memory_id, std::move(parent)),
        resources(LaunchArg<Args>(std::forward<Args>(args))...) {
        acquire(std::index_sequence_for<Args...> {});

        m_runtime.submit(m_requisition);
    }

    std::tuple<typename LaunchArg<Args>::resolve_type...> resolve() {
        if (m_requisition.stream().is_null()) {
            m_runtime.synchronize(m_requisition.dependencies());
        } else {
            m_requisition.stream().wait_on_events(m_requisition.dependencies());
        }

        return resolve(std::index_sequence_for<Args...> {});
    }

    void release(DeviceEventSet deps) {
        release_impl(std::index_sequence_for<Args...> {});
        m_runtime.release(m_requisition, deps);
    }

    ~Scope() {
        release({});
    }

  private:
    template<size_t... Is>
    void acquire(std::index_sequence<Is...>) {
        (std::get<Is>(resources).acquire(m_runtime, m_requisition), ...);
    }

    template<size_t... Is>
    std::tuple<typename LaunchArg<Args>::resolve_type...> resolve(std::index_sequence<Is...>) {
        return std::make_tuple(std::get<Is>(resources).resolve(m_runtime, m_requisition)...);
    }

    template<size_t... Is>
    void release_impl(std::index_sequence<Is...>) {
        (std::get<Is>(resources).release(m_runtime, m_requisition), ...);
    }

    Runtime m_runtime;
    Requisition m_requisition;
    std::tuple<LaunchArg<Args>...> resources;
};

}  // namespace kmm
