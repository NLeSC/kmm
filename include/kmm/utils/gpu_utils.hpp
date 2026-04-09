#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kmm/core/backends.hpp"
#include "kmm/utils/macros.hpp"

#define KMM_GPU_CHECK(...)                                                        \
    do {                                                                          \
        auto __code = (__VA_ARGS__);                                              \
        if (__code != decltype(__code)(0)) {                                      \
            ::kmm::gpu_throw_exception(__code, __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                         \
    } while (0);

namespace kmm {

void gpu_throw_exception(g_result_t result, const char* file, int line, const char* expression);
void gpu_throw_exception(gpu_error_t result, const char* file, int line, const char* expression);
void gpu_throw_exception(blas_status_t result, const char* file, int line, const char* expression);

class GPUException: public std::exception {
  public:
    GPUException(std::string message = {}) : m_message(std::move(message)) {}

    const char* what() const noexcept override {
        return m_message.c_str();
    }

  protected:
    std::string m_message;
};

class GPUDriverException: public GPUException {
  public:
    GPUDriverException(const std::string& message, g_result_t result);
    GPUDriverException(const char* message, g_result_t result) :
        GPUDriverException(std::string(message), result) {}
    g_result_t status;
};

class GPURuntimeException: public GPUException {
  public:
    GPURuntimeException(const std::string& message, gpu_error_t result);
    gpu_error_t status;
};

class BlasException: public GPUException {
  public:
    BlasException(const std::string& message, blas_status_t result);
    blas_status_t status;
};

/**
 * Returns the available devices as a list of `device`s.
 */
std::vector<g_device_t> get_gpu_devices();

/**
 * If the given address points to memory allocation that has been allocated on a GPU, then
 * this function returns the device ordinal as a `device`. If the address points ot an invalid
 * memory location or a non-GPU buffer, then it returns `std::nullopt`.
 */
std::optional<g_device_t> get_gpu_device_by_address(const void* address);

class GPUContextHandle {
    GPUContextHandle() = delete;
    GPUContextHandle(g_context_t context, std::shared_ptr<void> lifetime);

  public:
    static GPUContextHandle create_context_for_device(g_device_t device);
    static GPUContextHandle retain_primary_context_for_device(g_device_t device);

    operator g_context_t() const {
        return m_context;
    }

  private:
    g_context_t m_context;
    std::shared_ptr<void> m_lifetime;
};

inline bool operator==(const GPUContextHandle& lhs, const GPUContextHandle& rhs) {
    return g_context_t(lhs) == g_context_t(rhs);
}

inline bool operator!=(const GPUContextHandle& lhs, const GPUContextHandle& rhs) {
    return !(lhs == rhs);
}

class GPUContextGuard {
    KMM_NOT_COPYABLE_OR_MOVABLE(GPUContextGuard)

  public:
    GPUContextGuard(GPUContextHandle context);
    ~GPUContextGuard();

  private:
    GPUContextHandle m_context;
};

inline g_device_ptr_t gpu_deviceptr_offset(g_device_ptr_t ptr, size_t size) {
    return reinterpret_cast<g_device_ptr_t>(reinterpret_cast<unsigned long long>(ptr) + size);
}

}  // namespace kmm
