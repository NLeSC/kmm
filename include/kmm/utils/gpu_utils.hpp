#pragma once

#include <cuda.h>
#include <functional>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmt/ostream.h"

#include "kmm/core/macros.hpp"

#define KMM_CUDA_CHECK(...)                                                        \
    do {                                                                           \
        auto __code = (__VA_ARGS__);                                               \
        if (KMM_UNLIKELY(__code != decltype(__code)(0))) {                         \
            ::kmm::cuda_throw_exception(__code, __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                          \
    } while (0)

namespace kmm {

/// \addtogroup utility
/// @{

void cuda_throw_exception(CUresult result, const char* file, int line, const char* expression);

class CUDAException: public std::exception {
  public:
    CUDAException(std::string message = {}) : m_message(std::move(message)) {}

    const char* what() const noexcept override {
        return m_message.c_str();
    }

  protected:
    std::string m_message;
};

class CUDAContextGuard {
    KMM_NOT_COPYABLE_OR_MOVABLE(CUDAContextGuard)

  public:
    CUDAContextGuard(CUcontext context);
    ~CUDAContextGuard();

  private:
    CUcontext m_context;
};

CUcontext context_from_stream(CUstream stream);

struct CUDAContextId {
    CUDAContextId(CUcontext context);

    bool operator==(const CUDAContextId& that) const noexcept {
        return m_id == that.m_id;
    }

    bool operator!=(const CUDAContextId& that) const noexcept {
        return !(*this == that);
    }

    unsigned long long get() const noexcept {
        return m_id;
    }

    friend std::ostream& operator<<(std::ostream&, const CUDAContextId& self);

  private:
    unsigned long long m_id;
};

class CUDAStreamRef;
class CUDAStream;

struct CUDAStreamId {
    CUDAStreamId(CUstream stream);
    CUDAStreamId(CUstream stream, CUcontext context);
    CUDAStreamId(const CUDAStreamRef& stream);
    CUDAStreamId(const CUDAStream& stream);

    unsigned long long get() const noexcept {
        return m_id;
    }

    const CUDAContextId& context() const noexcept {
        return m_context_id;
    }

    bool operator==(const CUDAStreamId& that) const noexcept {
        return m_id == that.m_id && m_context_id == that.m_context_id;
    }

    bool operator!=(const CUDAStreamId& that) const noexcept {
        return !(*this == that);
    }

    friend std::ostream& operator<<(std::ostream&, const CUDAStreamId& self);

  private:
    CUDAContextId m_context_id;
    unsigned long long m_id;
};

class CUDAStreamRef {
  public:
    CUDAStreamRef(CUstream);

    CUcontext context() const noexcept {
        return m_context;
    }

    CUstream stream() const noexcept {
        return m_stream;
    }

    CUDAStreamId stream_id() const noexcept {
        return m_stream_id;
    }

    operator CUstream() const noexcept {
        return m_stream;
    }

    friend std::ostream& operator<<(std::ostream&, const CUDAStreamRef& self);

  private:
    CUcontext m_context = nullptr;
    CUstream m_stream = nullptr;
    CUDAStreamId m_stream_id;
};

class CUDAStream {
  public:
    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

    explicit CUDAStream(CUcontext context, unsigned int flags = CU_STREAM_NON_BLOCKING);
    ~CUDAStream();

    CUDAStream(CUDAStream&& that) noexcept : m_stream(that.m_stream) {
        that.m_stream = nullptr;
    }

    CUDAStream& operator=(CUDAStream&& that) noexcept {
        if (this != &that) {
            destroy();
            m_stream = that.m_stream;
            that.m_stream = nullptr;
        }

        return *this;
    }

    CUstream get() const noexcept {
        return m_stream;
    }

    operator CUstream() const noexcept {
        return m_stream;
    }

    operator CUDAStreamRef() const noexcept {
        return m_stream;
    }

    friend std::ostream& operator<<(std::ostream&, const CUDAStream& self);

  private:
    void destroy() noexcept;

    CUstream m_stream = nullptr;
};

/// @}

}  // namespace kmm

template<>
struct std::hash<kmm::CUDAContextId> {
    size_t operator()(const kmm::CUDAContextId& id) const noexcept {
        return std::hash<unsigned long long> {}(id.get());
    }
};

template<>
struct std::hash<kmm::CUDAStreamId> {
    size_t operator()(const kmm::CUDAStreamId& id) const noexcept {
        return std::hash<unsigned long long> {}(id.get());
    }
};

template<>
struct fmt::formatter<kmm::CUDAStream>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::CUDAStreamRef>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::CUDAStreamId>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::CUDAContextId>: fmt::ostream_formatter {};