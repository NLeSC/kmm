#pragma once

#include <ostream>
#include <utility>

#include "fmt/ostream.h"

#include "kmm/runtime/device_event_registry.hpp"

namespace kmm {

class DeviceStream {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceStream)

  public:
    DeviceStream() = default;

    DeviceStream(DeviceEventRegistry registry, DeviceStreamId stream_id) :
        m_manager(std::move(registry)),
        m_stream_id(stream_id),
        m_stream(m_manager.get(stream_id)),
        m_context(m_manager.context(stream_id)) {}

    DeviceStreamId id() const noexcept {
        return m_stream_id;
    }

    g_context_t context() const noexcept {
        return m_context;
    }

    operator g_stream_t() const noexcept {
        return m_stream;
    }

    bool is_null() const noexcept {
        return m_stream_id.is_null();
    }

    bool preceded_by(const DeviceEventSet& deps) const noexcept {
        return m_manager.precedes(deps, m_stream_id);
    }

    void wait_on_event(const DeviceEvent& dep) const {
        m_manager.wait_on_event(m_stream_id, dep);
    }

    void wait_on_event(const DeviceEventSet& deps) const {
        m_manager.wait_on_event(m_stream_id, deps);
    }

    void wait_on_default_stream() const {
        m_manager.wait_on_default_stream(m_stream_id);
    }

    void synchronize() const {
        m_manager.synchronize(m_stream_id);
    }

    DeviceEvent record_event() const {
        return m_manager.record(m_stream_id);
    }

    /**
     * Waits for `deps`, calls `fun(stream)` to submit work onto the stream, then records an event.
     */
    template<typename F>
    DeviceEvent submit(const DeviceEventSet& deps, F&& fun) const {
        return m_manager.submit(m_stream_id, deps, std::forward<F>(fun));
    }

    friend std::ostream& operator<<(std::ostream& stream, const DeviceStream& e) {
        return stream << e.m_stream_id;
    }

  private:
    DeviceEventRegistry m_manager;
    DeviceStreamId m_stream_id;
    g_stream_t m_stream = nullptr;
    g_context_t m_context = nullptr;
};

}  // namespace kmm

template<>
struct fmt::formatter<kmm::DeviceStream>: fmt::ostream_formatter {};
