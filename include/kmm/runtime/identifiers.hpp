#pragma once

#include <functional>
#include <iostream>

#include "fmt/ostream.h"

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/utils/hash_utils.hpp"

namespace kmm {

static constexpr size_t MAX_DEVICES = 4;

class DeviceId {
  public:
    KMM_INLINE constexpr explicit DeviceId(size_t id) : m_id(static_cast<uint8_t>(id)) {
        if (m_id >= MAX_DEVICES) {
            throw std::runtime_error("Device id out of range");
        }
    }

    KMM_INLINE size_t get() const noexcept {
        KMM_UNSAFE_ASSUME(m_id < MAX_DEVICES);
        return m_id;
    }

    KMM_INLINE constexpr bool operator==(const DeviceId& that) const noexcept {
        return m_id == that.m_id;
    }

    KMM_INLINE constexpr bool operator!=(const DeviceId& that) const noexcept {
        return !(*this == that);
    }

    friend std::ostream& operator<<(std::ostream& stream, const DeviceId& e);

  private:
    uint8_t m_id;
};

class BufferId {
  public:
    KMM_INLINE constexpr explicit BufferId(uint64_t id) : m_id(id) {}

    KMM_INLINE constexpr uint64_t get() const noexcept {
        return m_id;
    }

    KMM_INLINE constexpr bool operator==(const BufferId& that) const noexcept {
        return m_id == that.m_id;
    }

    KMM_INLINE constexpr bool operator!=(const BufferId& that) const noexcept {
        return !(*this == that);
    }

    friend std::ostream& operator<<(std::ostream& stream, const BufferId& e);

  private:
    uint64_t m_id;
};

class MemoryId {
  public:
    enum struct Kind { Host, Device };

    KMM_INLINE static constexpr MemoryId host() {
        return MemoryId {Kind::Host, DeviceId(0)};
    }

    KMM_INLINE static constexpr MemoryId device(DeviceId id) {
        return MemoryId {Kind::Device, id};
    }

    KMM_INLINE
    bool is_host() const noexcept {
        return m_kind == Kind::Host;
    }

    KMM_INLINE
    bool is_device() const noexcept {
        return !is_host();
    }

    KMM_INLINE
    DeviceId as_device() const noexcept {
        KMM_ASSERT(is_device());
        return m_device_id;
    }

    KMM_INLINE constexpr bool operator==(const MemoryId& that) const noexcept {
        return m_kind == that.m_kind && (m_kind != Kind::Device || m_device_id == that.m_device_id);
    }

    KMM_INLINE constexpr bool operator!=(const MemoryId& that) const noexcept {
        return !(*this == that);
    }

    KMM_INLINE constexpr bool operator<(const MemoryId& that) const noexcept {
        if (m_kind == Kind::Device && that.m_kind == Kind::Device) {
            return m_device_id.get() < that.m_device_id.get();
        } else {
            return m_kind < that.m_kind;
        }
    }

    KMM_INLINE constexpr bool operator>(const MemoryId& that) const noexcept {
        return that < *this;
    }

    KMM_INLINE constexpr bool operator<=(const MemoryId& that) const noexcept {
        return *this < that || *this == that;
    }

    KMM_INLINE constexpr bool operator>=(const MemoryId& that) const noexcept {
        return that <= *this;
    }

    friend std::ostream& operator<<(std::ostream& stream, const MemoryId& e);

  private:
    constexpr MemoryId(Kind kind, DeviceId device_id) : m_kind(kind), m_device_id(device_id) {}

    Kind m_kind;
    DeviceId m_device_id;
};

}  // namespace kmm

template<>
struct std::hash<kmm::BufferId> {
    size_t operator()(kmm::BufferId val) const noexcept {
        return ::kmm::hash_fields(val.get());
    }
};

template<>
struct std::hash<kmm::DeviceId> {
    size_t operator()(kmm::DeviceId val) const noexcept {
        return ::kmm::hash_fields(val.get());
    }
};

template<>
struct std::hash<kmm::MemoryId> {
    size_t operator()(kmm::MemoryId val) const noexcept {
        return val.is_device() ? val.as_device().get() : size_t(-1);
    }
};

template<>
struct fmt::formatter<kmm::DeviceId>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::BufferId>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::MemoryId>: fmt::ostream_formatter {};