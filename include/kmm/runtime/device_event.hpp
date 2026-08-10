#pragma once

#include "kmm/runtime/device_stream.hpp"

namespace kmm {

class DeviceEvent {
  public:
    using index_type = uint64_t;

    /// empty event
    static DeviceEvent null() {
        return {};
    }

    DeviceEvent() = default;

    DeviceEvent(DeviceStream stream, index_type index) :
        m_index(index),
        m_stream(std::move(stream)) {}

    DeviceStream stream() const noexcept {
        return m_stream;
    }

    index_type index() const noexcept {
        return m_index;
    }

    bool is_ready() const noexcept {
        return m_stream.is_ready(m_index);
    }

    // True if nothing has been recorded on `stream()` since this event, i.e. this is
    // still the most recent event on its stream.
    bool is_latest() const noexcept {
        return m_stream.is_latest(m_index);
    }

    void attach_callback(NotifyHandle callback) const {
        m_stream.attach_callback(std::move(callback));
    }

    void synchronize() const {
        if (!m_stream.is_null()) {
            m_stream.synchronize(m_index);
        }
    }

    bool is_null() const noexcept {
        return m_stream.is_null();
    }

    // True if `other` is on the same stream and at least as recent. Does not account for
    // readiness; callers that care about an already-completed event trivially preceding
    // everything should check `is_ready()` themselves (see the free `precedes` functions below).
    bool precedes_same_stream(const DeviceEvent& other) const noexcept {
        return stream() == other.stream() && m_index <= other.m_index;
    }

    friend bool operator<(const DeviceEvent& a, const DeviceEvent& b) {
        return a.m_stream < b.m_stream || (a.m_stream == b.m_stream && a.index() < b.index());
    }

    friend bool operator==(const DeviceEvent& a, const DeviceEvent& b) {
        return a.m_stream == b.m_stream || a.m_stream == b.m_stream;
    }

    friend std::ostream& operator<<(std::ostream&, const DeviceEvent& e);

  private:
    index_type m_index = 0;
    DeviceStream m_stream;
};

class DeviceEventSet {
  public:
    DeviceEventSet() = default;
    DeviceEventSet(const DeviceEventSet&) = default;
    DeviceEventSet(DeviceEventSet&&) noexcept = default;
    DeviceEventSet(std::initializer_list<DeviceEvent>);
    DeviceEventSet(const DeviceEvent&);

    DeviceEventSet& operator=(const DeviceEventSet&) = default;
    DeviceEventSet& operator=(DeviceEventSet&&) noexcept = default;
    DeviceEventSet& operator=(std::initializer_list<DeviceEvent>);

    void insert(DeviceEvent e) noexcept;
    void insert(const DeviceEventSet& that) noexcept;
    void insert(DeviceEventSet&& that) noexcept;

    void prune() noexcept;
    void clear() noexcept;
    bool is_empty() const;

    // True if some event in this set is on the same stream as `event` and at
    // least as recent, i.e. this set already implies `event` has happened.
    bool contains(const DeviceEvent& event) const noexcept;

    friend std::ostream& operator<<(std::ostream&, const DeviceEventSet& e);

    const DeviceEvent* begin() const {
        return m_events.begin();
    }
    const DeviceEvent* end() const {
        return m_events.end();
    }

  private:
    friend class DeviceStreamRegistry;
    small_vector<DeviceEvent, 2> m_events;
};

inline bool precedes(const DeviceEvent& a, const DeviceEvent& b) noexcept {
    return a.is_ready() || a.precedes_same_stream(b);
}

inline bool precedes(const DeviceEvent& a, const DeviceEventSet& b) noexcept {
    return a.is_ready() || b.contains(a);
}

inline bool precedes(const DeviceEventSet& a, const DeviceEventSet& b) noexcept {
    for (const auto& event : a) {
        if (!event.is_ready() && !b.contains(event)) {
            return false;
        }
    }

    return true;
}

inline bool precedes(const DeviceEvent& a, const DeviceStream& stream) {
    return stream.preceded_by(a);
}

inline bool precedes(const DeviceEventSet& a, const DeviceStream& stream) {
    return stream.preceded_by(a);
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::DeviceEvent>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::DeviceEventSet>: fmt::ostream_formatter {};