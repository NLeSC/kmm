#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "kmm/api/context.hpp"
#include "kmm/api/resource_guard.hpp"
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

template<typename... Args>
class DeviceGuard;

class Device: public Context {
  public:
    explicit Device(Runtime runtime, DeviceId device_id, MemoryTransaction transaction = {});

    DeviceStream stream() const noexcept;

    DeviceId device_id() const noexcept {
        return m_device_id;
    }

    operator g_stream_t() const noexcept {
        return *m_stream;
    }

    /// Acquires `args` and calls `fun(stream, views...)`, where `stream` is this device's
    /// `GPUStream` and `views...` are the resolved views for `args...`.
    template<typename F, typename... Args>
    decltype(auto) submit(F&& fun, Args&&... args) {
        return access(std::forward<Args>(args)...).submit(std::forward<F>(fun));
    }

    template<typename... Args>
    DeviceGuard<Args...> access(Args&&... args) {
        return DeviceGuard<Args...>(
            runtime(),
            m_device_id,
            m_stream,
            transaction(),
            std::forward<Args>(args)...
        );
    }

    /// Shorthand for `access(args...).parallel_for(shape, fun)`: applies `fun` to every point of
    /// the N-dimensional index space `shape`, one GPU thread per point.
    template<size_t N, typename F, typename... Args>
    decltype(auto) parallel_for(Shape<N> shape, F fun, Args&&... args) {
        return access(std::forward<Args>(args)...).parallel_for(shape, std::move(fun));
    }

    MemoryId affinity_memory_id() const noexcept override {
        return MemoryId::device(m_device_id);
    }

  private:
    DeviceId m_device_id;
    std::shared_ptr<GPUStreamOwner> m_stream;
};

template<typename... Args>
class DeviceGuard {
  public:
    DeviceGuard(
        Runtime& runtime,
        DeviceId device_id,
        std::shared_ptr<GPUStreamOwner> stream,
        MemoryTransaction parent,
        Args... args
    ) :
        m_device_id(device_id),
        m_stream(stream),
        m_guard(runtime, MemoryId::device(device_id), *stream, parent, args...) {
        auto& registry = runtime.event_registry();
        m_stream_id = registry.lookup_or_register_stream(*m_stream);
        registry.wait_on_event(m_stream_id, m_guard.dependencies());
    }

    ~DeviceGuard() {
        auto event = m_guard.runtime().event_registry().record(m_stream_id);
        m_guard.release(event);
    }

    Device context() const {
        return Device(m_guard.runtime(), m_device_id, m_guard.transaction());
    }

    DeviceId device_id() const noexcept {
        return m_device_id;
    }

    DeviceStreamId stream_id() const noexcept {
        return m_stream_id;
    }

    GPUStreamRef stream() const noexcept {
        return *m_stream;
    }

    template<size_t I>
    decltype(auto) get() {
        return m_guard.template get<I>();
    }

    template<typename F>
    void submit(F fun) {
        m_guard.apply(std::move(fun), context());
    }

    template<typename F>
    void operator>>(F fun) {
        submit(std::move(fun));
    }

    /// Shorthand for `submit(ParallelFor(shape, fun))`: applies `fun` to every point of the
    /// N-dimensional index space `shape`, one GPU thread per point, using the views already
    /// acquired by this guard.
    template<size_t N, typename F>
    void parallel_for(Shape<N> shape, F fun) {
        submit(ParallelFor<F, N>(shape, std::move(fun)));
    }

  private:
    DeviceId m_device_id;
    DeviceStreamId m_stream_id;
    std::shared_ptr<GPUStreamOwner> m_stream;
    ResourceGuard<Args...> m_guard;
};

}  // namespace kmm
