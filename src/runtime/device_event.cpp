#include <algorithm>
#include <utility>

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_event_registry.hpp"

namespace kmm {

DeviceEventSet::DeviceEventSet(std::initializer_list<DeviceEvent> list) {
    *this = list;
}

DeviceEventSet::DeviceEventSet(const DeviceEvent& event) {
    insert(event);
}

DeviceEventSet& DeviceEventSet::operator=(std::initializer_list<DeviceEvent> list) {
    m_events.resize(list.size());
    m_events.clear();

    for (auto e : list) {
        insert(e);
    }

    return *this;
}

void DeviceEventSet::insert(DeviceEvent e) noexcept {
    if (e.is_null()) {
        return;
    }

    bool found = false;

    for (size_t i = 0; i < m_events.size(); i++) {
        if (m_events[i].stream() == e.stream()) {
            m_events[i] = std::max(m_events[i], e);
            found = true;
        }
    }

    if (KMM_UNLIKELY(!found)) {
        KMM_ASSERT(m_events.try_push_back(e));
    }
}

void DeviceEventSet::insert(const DeviceEventSet& that) noexcept {
    size_t n = m_events.size();

    for (const auto& e : that.m_events) {
        bool found = false;

        for (size_t i = 0; i < n; i++) {
            if (m_events[i].stream() == e.stream()) {
                m_events[i] = std::max(m_events[i], e);
                found = true;
            }
        }

        if (KMM_UNLIKELY(!found)) {
            KMM_ASSERT(m_events.try_push_back(e));
        }
    }
}

void DeviceEventSet::insert(DeviceEventSet&& that) noexcept {
    if (m_events.is_empty()) {
        m_events = std::move(that.m_events);
    } else {
        insert(that);
    }

    that.clear();
}

void DeviceEventSet::prune(const DeviceEventRegistry& registry) noexcept {
    size_t index = 0;

    while (true) {
        if (index >= m_events.size()) {
            return;
        }

        if (registry.is_ready(m_events[index])) {
            break;
        }

        index++;
    }

    size_t new_size = m_events.size() - 1;
    std::swap(m_events[index], m_events[new_size]);

    while (index < new_size) {
        if (!registry.is_ready(m_events[index])) {
            index++;
        } else {
            new_size--;
            std::swap(m_events[index], m_events[new_size]);
        }
    }

    m_events.truncate(new_size);
}

void DeviceEventSet::clear() noexcept {
    m_events.clear();
}

bool DeviceEventSet::is_empty() const noexcept {
    return m_events.is_empty();
}

bool DeviceEventSet::contains(const DeviceEvent& event) const noexcept {
    if (event.is_null()) {
        return true;
    }

    for (const auto& e : m_events) {
        if (event.precedes(e)) {
            return true;
        }
    }

    return false;
}

bool DeviceEventSet::contains(const DeviceEventSet& events) const noexcept {
    for (const auto& e : events.m_events) {
        if (!contains(e)) {
            return false;
        }
    }

    return true;
}

DeviceEvent DeviceEventSet::find(DeviceStreamId stream_id) const noexcept {
    for (const auto& e : m_events) {
        if (e.stream() == stream_id) {
            return e;
        }
    }

    return DeviceEvent {};
}

std::ostream& operator<<(std::ostream& stream, const DeviceStreamId& e) {
    if (e.is_null()) {
        return stream << "<none>";
    } else {
        return stream << e.get();
    }
}

std::ostream& operator<<(std::ostream& stream, const DeviceEvent& e) {
    if (e.is_null()) {
        return stream << "<none>";
    } else {
        return stream << e.stream() << ":" << e.index();
    }
}

std::ostream& operator<<(std::ostream& stream, const DeviceEventSet& e) {
    std::vector<DeviceEvent> events = {e.m_events.begin(), e.m_events.end()};
    std::sort(events.begin(), events.end());

    stream << "{";
    bool is_first = true;

    for (const auto& item : events) {
        if (!is_first) {
            stream << ", ";
        }

        is_first = false;
        stream << item;
    }

    return stream << "}";
}

}  // namespace kmm
