#pragma once

#include <functional>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmt/ostream.h"

#include "kmm/core/macros.hpp"
#include "kmm/utils/gpu_api.hpp"

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

void gpu_throw_exception(GPUResult result, const char* file, int line, const char* expression);

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
    GPUContextGuard(GPUContext context);
    ~GPUContextGuard();

  private:
    GPUContext m_context;
};

GPUContext context_from_stream(GPUStream stream);

struct GPUContextId {
    GPUContextId(GPUContext context);

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
    GPUStreamId(GPUStream stream);
    GPUStreamId(GPUStream stream, GPUContext context);
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
    GPUStreamRef(GPUStream);

    GPUContext context() const noexcept {
        return m_context;
    }

    GPUStream stream() const noexcept {
        return m_stream;
    }

    GPUStreamId stream_id() const noexcept {
        return m_stream_id;
    }

    operator GPUStream() const noexcept {
        return m_stream;
    }

    friend std::ostream& operator<<(std::ostream&, const GPUStreamRef& self);

  private:
    GPUContext m_context = nullptr;
    GPUStream m_stream = nullptr;
    GPUStreamId m_stream_id;
};

class GPUStreamOwner {
  public:
    GPUStreamOwner(const GPUStreamOwner&) = delete;
    GPUStreamOwner& operator=(const GPUStreamOwner&) = delete;

    explicit GPUStreamOwner(GPUContext context, unsigned int flags = GPU_STREAM_DEFAULT_FLAGS);
    ~GPUStreamOwner();

    GPUStreamOwner(GPUStreamOwner&& that) noexcept : m_stream(that.m_stream) {
        that.m_stream = nullptr;
    }

    GPUStreamOwner& operator=(GPUStreamOwner&& that) noexcept {
        std::swap(m_stream, that.m_stream);
        return *this;
    }

    GPUStream get() const noexcept {
        return m_stream;
    }

    operator GPUStream() const noexcept {
        return m_stream;
    }

    operator GPUStreamRef() const noexcept {
        return m_stream;
    }

    friend std::ostream& operator<<(std::ostream&, const GPUStreamOwner& self);

  private:
    void destroy() noexcept;

    GPUStream m_stream = nullptr;
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
