#pragma once

#include "fmt/format.h"

#include "kmm/utils/hash_utils.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

static constexpr uint64_t MAX_DEVICE_STREAMS = 256;
static constexpr uint64_t MAX_DEVICE_EVENTS = (~uint64_t(0)) / MAX_DEVICE_STREAMS - 1;
class DeviceEventRegistry;
class PrecedenceVector;

class DeviceStreamId {
    static constexpr uint64_t INVALID_INDEX = ~uint64_t(0);

  public:
    DeviceStreamId() noexcept = default;

    explicit DeviceStreamId(uint64_t index) : m_index(index) {
        KMM_ASSERT(index < MAX_DEVICE_STREAMS);
    }

    static DeviceStreamId null() noexcept {
        return {};
    }

    uint64_t get() const noexcept {
        KMM_UNSAFE_ASSUME(m_index < MAX_DEVICE_STREAMS);
        return m_index;
    }

    bool is_null() const noexcept {
        return m_index == INVALID_INDEX;
    }

    friend std::ostream& operator<<(std::ostream& stream, const DeviceStreamId& e);

    friend bool operator==(const DeviceStreamId& a, const DeviceStreamId& b) {
        return a.get() == b.get();
    }

    friend bool operator!=(const DeviceStreamId& a, const DeviceStreamId& b) {
        return !(a == b);
    }

  private:
    uint64_t m_index = INVALID_INDEX;
};

class DeviceEvent {
  public:
    DeviceEvent() noexcept = default;

    DeviceEvent(DeviceStreamId stream_id, uint64_t event_id) {
        if (!stream_id.is_null()) {
            KMM_ASSERT(stream_id.get() < MAX_DEVICE_STREAMS);
            m_event_and_stream_index = stream_id.get() + event_id * MAX_DEVICE_STREAMS;
        }
    }

    static DeviceEvent null() noexcept {
        return {};
    }

    DeviceStreamId stream() const noexcept {
        return DeviceStreamId(m_event_and_stream_index % MAX_DEVICE_STREAMS);
    }

    uint64_t index() const {
        return m_event_and_stream_index / MAX_DEVICE_STREAMS;
    }

    size_t hash() const {
        return m_event_and_stream_index;
    }

    bool is_null() const noexcept {
        return m_event_and_stream_index == 0;
    }

    bool precedes(const DeviceEvent& that) const noexcept {
        return stream() == that.stream()
            && this->m_event_and_stream_index <= that.m_event_and_stream_index;
    }

    friend std::ostream& operator<<(std::ostream& stream, const DeviceEvent& e);

    friend bool operator==(const DeviceEvent& a, const DeviceEvent& b) {
        return a.m_event_and_stream_index == b.m_event_and_stream_index;
    }

    friend bool operator<(const DeviceEvent& a, const DeviceEvent& b) {
        return a.m_event_and_stream_index < b.m_event_and_stream_index;
    }

    friend bool operator<=(const DeviceEvent& a, const DeviceEvent& b) {
        return a.m_event_and_stream_index <= b.m_event_and_stream_index;
    }

    friend bool operator!=(const DeviceEvent& a, const DeviceEvent& b) {
        return !(a == b);
    }

    friend bool operator>(const DeviceEvent& a, const DeviceEvent& b) {
        return b < a;
    }

    friend bool operator>=(const DeviceEvent& a, const DeviceEvent& b) {
        return b <= a;
    }

  private:
    uint64_t m_event_and_stream_index = 0;
};

class DeviceEventSet {
  public:
    DeviceEventSet() noexcept = default;
    DeviceEventSet(const DeviceEvent& event);
    DeviceEventSet(const DeviceEventSet& events) = default;
    DeviceEventSet(DeviceEventSet&& events) noexcept = default;
    DeviceEventSet(std::initializer_list<DeviceEvent> list);

    DeviceEventSet& operator=(const DeviceEventSet& that) = default;
    DeviceEventSet& operator=(DeviceEventSet&& that) noexcept = default;
    DeviceEventSet& operator=(std::initializer_list<DeviceEvent> list);

    void insert(DeviceEvent event) noexcept;
    void insert(const DeviceEventSet& events) noexcept;
    void insert(DeviceEventSet&& events) noexcept;
    void prune(const DeviceEventRegistry& registry) noexcept;
    void clear() noexcept;
    bool is_empty() const noexcept;
    bool contains(const DeviceEvent& event) const noexcept;
    bool contains(const DeviceEventSet& events) const noexcept;
    DeviceEvent find(DeviceStreamId stream_id) const noexcept;

    const DeviceEvent* begin() const noexcept {
        return m_events.begin();
    }
    const DeviceEvent* end() const noexcept {
        return m_events.end();
    }

    friend std::ostream& operator<<(std::ostream& stream, const DeviceEventSet& e);

    // from device_event_registry.cpp
    friend class PrecedenceVector;

  private:
    small_vector<DeviceEvent, 4> m_events;
};

}  // namespace kmm

template<>
struct std::hash<kmm::DeviceStreamId> {
    size_t operator()(const kmm::DeviceStreamId& id) const noexcept {
        return id.get();
    }
};

template<>
struct std::hash<kmm::DeviceEvent> {
    size_t operator()(const kmm::DeviceEvent& id) const noexcept {
        return id.hash();
    }
};

template<>
struct fmt::formatter<kmm::DeviceStreamId>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::DeviceEvent>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::DeviceEventSet>: fmt::ostream_formatter {};