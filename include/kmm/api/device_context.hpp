#pragma once

#include <optional>
#include <utility>

#include "kmm/api/context.hpp"
#include "kmm/core/shape.hpp"
#include "kmm/runtime/device_data_streams.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/utils/gpu_utils.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

template<typename F, size_t N>
class ParallelFor;

class DeviceContext: public Context {
  public:
    explicit DeviceContext(Runtime runtime, DeviceId device_id, MemoryTransaction transaction = {});

    DeviceId device_id() const noexcept {
        return m_device_id;
    }

    DeviceStream stream() const noexcept;

    operator CUstream() const noexcept {
        return *m_stream;
    }

    template<typename... Args>
    Scope<DeviceContext, Args...> scope(Args&&... args) {
        auto scope = std::make_unique<ScopeImpl<Args...>>(
            m_runtime,
            MemoryId::device(m_device_id),
            *m_stream,
            m_transaction,
            std::forward<Args>(args)...
        );
        auto context =
            std::make_unique<DeviceContext>(m_runtime, m_device_id, scope->transaction());
        return Scope<DeviceContext, Args...>(std::move(scope), std::move(context));
    }

    /// Shorthand for `scope(args...).apply(fun)`.
    template<typename Launcher, typename... Args>
    decltype(auto) launch(Launcher&& launcher, Args&&... args) {
        return scope(std::forward<Args>(args)...).apply(std::forward<Launcher>(launcher));
    }

    /// Shorthand for `launch(ParallelFor(shape, fun), args...)`: applies `fun` to every point of
    /// the N-dimensional index space `shape`, one GPU thread per point.
    template<size_t N, typename F, typename... Args>
    decltype(auto) parallel_for(Shape<N> shape, F fun, Args&&... args) {
        return launch(ParallelFor<F, N>(shape, std::move(fun)), std::forward<Args>(args)...);
    }

    MemoryId affinity_memory_id() const noexcept override {
        return MemoryId::device(m_device_id);
    }

  private:
    DeviceId m_device_id;
    std::shared_ptr<CUDAStream> m_stream;
};

template<>
struct scope_invoke<DeviceContext> {
    template<typename ScopeT, typename F>
    void operator()(DeviceContext& context, ScopeT& scope, DeviceEventSet& deps, F&& fun) const {
        auto strm = context.stream();
        auto& events = context.runtime().event_registry();

        CUDAContextGuard guard(strm.context());
        strm.wait_on_event(scope.dependencies());

        scope.apply(std::forward<F>(fun), context);

        events.wait_on_default_stream(strm.id());
        deps.insert(events.record(strm.id()));
    }
};

}  // namespace kmm
