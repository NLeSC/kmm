#include "fmt/format.h"

#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

void gpu_throw_exception(g_result_t result, const char* file, int line, const char* expression) {
    const char* name = "UNKNOWN_ERROR";
    const char* description = "unknown error";
    g_get_error_name(result, &name);
    g_get_error_string(result, &description);

    throw GPUException(
        fmt::format("GPU error: {} ({}) at {}:{}: {}", name, description, file, line, expression)
    );
}

GPUContextGuard::GPUContextGuard(g_context_t context) : m_context(context) {
    KMM_GPU_CHECK(g_ctx_push_current(context));
}

GPUContextGuard::~GPUContextGuard() {
    // Destructors must not throw, so the pop result is discarded rather than checked.
    g_context_t popped = nullptr;
    g_ctx_pop_current(&popped);
}

GPUContextId::GPUContextId(g_context_t context) {
#if defined(KMM_USE_HIP)
    // HIP has no equivalent of `cuCtxGetId`. The context handle itself is
    // already a stable, unique identity, so it is used directly.
    m_id = reinterpret_cast<unsigned long long>(context);
#else
    KMM_GPU_CHECK(cuCtxGetId(context, &m_id));
#endif
}

g_context_t context_from_stream(g_stream_t stream) {
    g_context_t context;
#if defined(KMM_USE_HIP)
    // HIP has no equivalent of `cuStreamGetCtx`. Instead, resolve the device
    // the stream was created on and reuse its primary context, which is what
    // `SystemInfo` already treats as "the" context for that device.
    GPUDevice device;
    KMM_GPU_CHECK(gpuStreamGetDevice(stream, &device));
    KMM_GPU_CHECK(gpuDevicePrimaryCtxRetain(&context, device));
    gpuDevicePrimaryCtxRelease(device);
#else
    KMM_GPU_CHECK(g_stream_get_ctx(stream, &context));
#endif
    return context;
}

GPUStreamId::GPUStreamId(g_stream_t stream) : GPUStreamId(stream, context_from_stream(stream)) {}

GPUStreamId::GPUStreamId(g_stream_t stream, g_context_t context) : m_context_id(context) {
#if defined(KMM_USE_HIP)
    // HIP has no equivalent of `cuStreamGetId`. The stream handle itself is
    // already a stable, unique identity, so it is used directly.
    m_id = reinterpret_cast<unsigned long long>(stream);
#else
    KMM_GPU_CHECK(cuStreamGetId(stream, &m_id));
#endif
}

GPUStreamId::GPUStreamId(const GPUStreamRef& stream) : GPUStreamId(stream.stream_id()) {}

GPUStreamId::GPUStreamId(const GPUStreamOwner& stream) :
    GPUStreamId(static_cast<GPUStreamRef>(stream)) {}

GPUStreamRef::GPUStreamRef(g_stream_t stream) :
    m_context(context_from_stream(stream)),
    m_stream(stream),
    m_stream_id(stream, m_context) {}

GPUStreamOwner::GPUStreamOwner(g_context_t context, unsigned int flags) :
    m_stream([&]() {
        g_stream_t result;
        GPUContextGuard guard(context);
        KMM_GPU_CHECK(g_stream_create(&result, flags));
        return result;
    }()) {}

GPUStreamOwner::~GPUStreamOwner() {
    destroy();
}

void GPUStreamOwner::destroy() noexcept {
    if (m_stream != nullptr) {
        g_stream_destroy(m_stream);
        m_stream = nullptr;
    }
}

std::ostream& operator<<(std::ostream& stream, const GPUStreamOwner& self) {
    if (self.m_stream == nullptr) {
        return stream << "GPU-stream: none";
    }

    return stream << GPUStreamRef(self.m_stream);
}

std::ostream& operator<<(std::ostream& stream, const GPUStreamRef& self) {
    return stream << self.m_stream_id;
}

std::ostream& operator<<(std::ostream& stream, const GPUStreamId& self) {
    return stream << "GPU-stream:" << self.m_id;
}

std::ostream& operator<<(std::ostream& stream, const GPUContextId& self) {
    return stream << "GPU-context:" << self.m_id;
}

}  // namespace kmm
