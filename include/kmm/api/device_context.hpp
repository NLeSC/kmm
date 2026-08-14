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

// Defined in `kmm/api/parallel_for.hpp`. Forward-declared here so `DeviceContext::parallel_for`
// can build one without `device_context.hpp` having to include (and thus depend on) it back.
template<typename F, size_t N>
class ParallelFor;

class DeviceContext: public Context {
  public:
    explicit DeviceContext(Runtime runtime, DeviceId device_id, DeviceEventSet deps = {});

    DeviceId device_id() const noexcept {
        return m_device_id;
    }

    DeviceStream stream() const noexcept;

    operator CUstream() const noexcept {
        return *m_stream;
    }

    template<typename... Args>
    Scope<DeviceContext, Args...> scope(Args&&... args) {
        return Scope<DeviceContext, Args...>(
            *this,
            MemoryId::device(m_device_id),
            transaction(),
            std::forward<Args>(args)...
        );
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
    template<typename, typename...>
    friend class Scope;

    template<typename...>
    friend struct ScopeImpl;

    // See `Context`'s protected rebind constructor.
    DeviceContext(const DeviceContext& parent, MemoryTransaction transaction) :
        Context(parent, std::move(transaction)),
        m_device_id(parent.m_device_id),
        m_stream(parent.m_stream) {}

    std::optional<CUDAStreamRef> submit_stream() noexcept;

    // Makes this device's CUDA context current on the calling thread for the duration of the
    // guard. `Scope::apply` calls this before invoking user code, since kernel launches and other
    // driver-API calls require the context to be active on whichever thread executes them.
    CUDAContextGuard activate() const {
        return CUDAContextGuard(runtime().system_info().device(m_device_id).context());
    }

    DeviceContext child(MemoryTransaction transaction) noexcept {
        return DeviceContext(*this, std::move(transaction));
    }

    DeviceId m_device_id;
    std::shared_ptr<CUDAStream> m_stream;
};

}  // namespace kmm
