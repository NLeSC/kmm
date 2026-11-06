#pragma once

#include <functional>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmt/ostream.h"

#include "kmm/core/macros.hpp"
#include "kmm/utils/backends.hpp"

#define KMM_GPU_CHECK(...)                                                        \
    do {                                                                          \
        auto __code = (__VA_ARGS__);                                              \
        if (KMM_UNLIKELY(__code != decltype(__code)(0))) {                        \
            ::kmm::gpu_throw_exception(__code, __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                         \
    } while (0)

namespace kmm {

/// \addtogroup utility
/// @{

void gpu_throw_exception(g_result_t result, const char* file, int line, const char* expression);

class GPUException: public std::exception {
  public:
    GPUException(std::string message = {}) : m_message(std::move(message)) {}

    const char* what() const noexcept override {
        return m_message.c_str();
    }

  protected:
    std::string m_message;
};

class GPUContextGuard {
    KMM_NOT_COPYABLE_OR_MOVABLE(GPUContextGuard)

  public:
    GPUContextGuard(g_context_t context);
    ~GPUContextGuard();

  private:
    g_context_t m_context;
};

g_context_t context_from_stream(g_stream_t stream);

struct GPUContextId {
    GPUContextId(g_context_t context);

    bool operator==(const GPUContextId& that) const noexcept {
        return m_id == that.m_id;
    }

    bool operator!=(const GPUContextId& that) const noexcept {
        return !(*this == that);
    }

    unsigned long long get() const noexcept {
        return m_id;
    }

    friend std::ostream& operator<<(std::ostream&, const GPUContextId& self);

  private:
    unsigned long long m_id;
};

class GPUStreamRef;
class GPUStreamOwner;

struct GPUStreamId {
    GPUStreamId(g_stream_t stream);
    GPUStreamId(g_stream_t stream, g_context_t context);
    GPUStreamId(const GPUStreamRef& stream);
    GPUStreamId(const GPUStreamOwner& stream);

    unsigned long long get() const noexcept {
        return m_id;
    }

    const GPUContextId& context() const noexcept {
        return m_context_id;
    }

    bool operator==(const GPUStreamId& that) const noexcept {
        return m_id == that.m_id && m_context_id == that.m_context_id;
    }

    bool operator!=(const GPUStreamId& that) const noexcept {
        return !(*this == that);
    }

    friend std::ostream& operator<<(std::ostream&, const GPUStreamId& self);

  private:
    GPUContextId m_context_id;
    unsigned long long m_id;
};

class GPUStreamRef {
  public:
    GPUStreamRef(g_stream_t);

    g_context_t context() const noexcept {
        return m_context;
    }

    g_stream_t stream() const noexcept {
        return m_stream;
    }

    GPUStreamId stream_id() const noexcept {
        return m_stream_id;
    }

    operator g_stream_t() const noexcept {
        return m_stream;
    }

    friend std::ostream& operator<<(std::ostream&, const GPUStreamRef& self);

  private:
    g_context_t m_context = nullptr;
    g_stream_t m_stream = nullptr;
    GPUStreamId m_stream_id;
};

class GPUStreamOwner {
  public:
    GPUStreamOwner(const GPUStreamOwner&) = delete;
    GPUStreamOwner& operator=(const GPUStreamOwner&) = delete;

    explicit GPUStreamOwner(g_context_t context, unsigned int flags = G_STREAM_NON_BLOCKING);
    ~GPUStreamOwner();

    GPUStreamOwner(GPUStreamOwner&& that) noexcept : m_stream(that.m_stream) {
        that.m_stream = nullptr;
    }

    GPUStreamOwner& operator=(GPUStreamOwner&& that) noexcept {
        std::swap(m_stream, that.m_stream);
        return *this;
    }

    g_stream_t get() const noexcept {
        return m_stream;
    }

    operator g_stream_t() const noexcept {
        return m_stream;
    }

    operator GPUStreamRef() const noexcept {
        return m_stream;
    }

    friend std::ostream& operator<<(std::ostream&, const GPUStreamOwner& self);

  private:
    void destroy() noexcept;

    g_stream_t m_stream = nullptr;
};

/// @}

}  // namespace kmm

template<>
struct std::hash<kmm::GPUContextId> {
    size_t operator()(const kmm::GPUContextId& id) const noexcept {
        return std::hash<unsigned long long> {}(id.get());
    }
};

template<>
struct std::hash<kmm::GPUStreamId> {
    size_t operator()(const kmm::GPUStreamId& id) const noexcept {
        return std::hash<unsigned long long> {}(id.get());
    }
};

template<>
struct fmt::formatter<kmm::GPUStreamOwner>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::GPUStreamRef>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::GPUStreamId>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::GPUContextId>: fmt::ostream_formatter {};
