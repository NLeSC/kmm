#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/utils/gpu_utils.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

void gpu_throw_exception(g_result_t result, const char* file, int line, const char* expression) {
    throw GPUDriverException(fmt::format("{} ({}:{})", expression, file, line), result);
}

#ifndef KMM_USE_HIP
void gpu_throw_exception(gpu_error_t result, const char* file, int line, const char* expression) {
    throw GPURuntimeException(fmt::format("{} ({}:{})", expression, file, line), result);
}
#endif

void gpu_throw_exception(blas_status_t result, const char* file, int line, const char* expression) {
    throw BlasException(fmt::format("{} ({}:{})", expression, file, line), result);
}

GPUDriverException::GPUDriverException(const std::string& message, g_result_t result) :
    status(result) {
    const char* name = "???";
    const char* description = "???";

    // Ignore the return code from these functions
    g_get_error_name(result, &name);
    g_get_error_string(result, &description);

    m_message = fmt::format("GPU driver error: {} ({}): {}", description, name, message);
}

GPURuntimeException::GPURuntimeException(const std::string& message, gpu_error_t result) :
    status(result) {
    const char* name = "???";
    const char* description = "???";

    // Ignore the return code from these functions
    name = gpu_get_error_name(result);
    description = gpu_get_error_string(result);

    m_message = fmt::format("GPU runtime error: {} ({}): {}", description, name, message);
}

BlasException::BlasException(const std::string& message, blas_status_t result) : status(result) {
    const char* name = blas_get_status_name(result);
    const char* description = blas_get_status_string(result);

    m_message = fmt::format("BLAS runtime error: {} ({}): {}", description, name, message);
}

GPUContextHandle::GPUContextHandle(g_context_t context, std::shared_ptr<void> lifetime) :
    m_context(context),
    m_lifetime(std::move(lifetime)) {}

std::vector<g_device_t> get_gpu_devices() {
    try {
        auto result = g_init(0);
        if (result == G_ERROR_NO_DEVICE) {
            return {};
        }

        if (result != G_SUCCESS) {
            throw GPUDriverException("gpuInit failed", result);
        }

        int count = 0;
        KMM_GPU_CHECK(g_device_get_count(&count));

        std::vector<g_device_t> devices {};
        for (int i = 0; i < count; i++) {
            g_device_t device;
            KMM_GPU_CHECK(g_device_get(&device, i));
            devices.push_back(device);
        }

        return devices;
    } catch (const GPUException& e) {
        spdlog::warn("ignored error while initializing: {}", e.what());
        return {};
    }
}

std::optional<g_device_t> get_gpu_device_by_address(const void* address) {
    int ordinal;
    g_memory_type_t memory_type;
    g_result_t result = g_pointer_get_attribute(
        &memory_type,
        G_POINTER_ATTRIBUTE_MEMORY_TYPE,
        g_device_ptr_t(address)
    );

    if (result == G_SUCCESS && memory_type == G_MEMORYTYPE_DEVICE) {
        result = g_pointer_get_attribute(
            &ordinal,
            G_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            g_device_ptr_t(address)
        );

        if (result == G_SUCCESS) {
            return g_device_t {ordinal};
        }
    }

    return std::nullopt;
}

GPUContextHandle GPUContextHandle::create_context_for_device(g_device_t device) {
    int flags = G_CTX_MAP_HOST;
    g_context_t context;
    KMM_GPU_CHECK(g_ctx_create(&context, flags, device));

    auto lifetime = std::shared_ptr<void>(nullptr, [=](const void* ignore) {
        KMM_ASSERT(g_ctx_destroy(context) == G_SUCCESS);
    });

    return {context, lifetime};
}

GPUContextHandle GPUContextHandle::retain_primary_context_for_device(g_device_t device) {
    g_context_t context;
    KMM_GPU_CHECK(g_device_primary_ctx_retain(&context, device));

    auto lifetime = std::shared_ptr<void>(nullptr, [=](const void* ignore) {
        KMM_ASSERT(g_device_primary_ctx_release(device) == G_SUCCESS);
    });

    return {context, lifetime};
}

GPUContextGuard::GPUContextGuard(GPUContextHandle context) : m_context(std::move(context)) {
    KMM_GPU_CHECK(g_ctx_push_current(m_context));
}

GPUContextGuard::~GPUContextGuard() {
    g_context_t previous;
    KMM_ASSERT(g_ctx_pop_current(&previous) == G_SUCCESS);
}

}  // namespace kmm
