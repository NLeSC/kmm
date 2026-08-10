#pragma once

#include <cuda.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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

struct cuda_context_id {
    cuda_context_id(CUcontext context);

    bool operator==(const cuda_context_id& that) const noexcept {
        return m_id == that.m_id;
    }

    bool operator!=(const cuda_context_id& that) const noexcept {
        return !(*this == that);
    }

    unsigned long long get() const noexcept {
        return m_id;
    }

  private:
    unsigned long long m_id;
};

class CUDAStream {
  public:
    explicit CUDAStream(CUcontext context, unsigned int flags = CU_STREAM_NON_BLOCKING);
    ~CUDAStream();

    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

    CUDAStream(CUDAStream&& that) noexcept : m_stream(std::exchange(that.m_stream, nullptr)) {}

    CUDAStream& operator=(CUDAStream&& that) noexcept {
        std::swap(m_stream, that.m_stream);
        return *this;
    }

    CUstream get() const noexcept {
        return m_stream;
    }

    operator CUstream() const noexcept {
        return m_stream;
    }

  private:
    CUstream m_stream = nullptr;
};

struct cuda_stream_id {
    cuda_stream_id(CUstream stream);

    bool operator==(const cuda_stream_id& that) const noexcept {
        return m_id == that.m_id;
    }

    bool operator!=(const cuda_stream_id& that) const noexcept {
        return !(*this == that);
    }

    unsigned long long get() const noexcept {
        return m_id;
    }

    const cuda_context_id& context() const noexcept {
        return m_context_id;
    }

  private:
    cuda_context_id m_context_id;
    unsigned long long m_id;
};

/// @}

}  // namespace kmm

template<>
struct std::hash<kmm::cuda_context_id> {
    size_t operator()(const kmm::cuda_context_id& id) const noexcept {
        return std::hash<unsigned long long> {}(id.get());
    }
};

template<>
struct std::hash<kmm::cuda_stream_id> {
    size_t operator()(const kmm::cuda_stream_id& id) const noexcept {
        return std::hash<unsigned long long> {}(id.get());
    }
};