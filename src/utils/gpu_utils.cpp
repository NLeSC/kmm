#include "fmt/format.h"

#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

void gpu_throw_exception(GPUResult result, const char* file, int line, const char* expression) {
    const char* name = "UNKNOWN_ERROR";
    const char* description = "unknown error";
    gpuGetErrorName(result, &name);
    gpuGetErrorString(result, &description);

    throw GPUException(
        fmt::format("GPU error: {} ({}) at {}:{}: {}", name, description, file, line, expression)
    );
}

GPUContextGuard::GPUContextGuard(GPUContext context) : m_context(context) {
    KMM_GPU_CHECK(gpuCtxPushCurrent(context));
}

GPUContextGuard::~GPUContextGuard() {
    // Destructors must not throw, so the pop result is discarded rather than checked.
    GPUContext popped = nullptr;
    gpuCtxPopCurrent(&popped);
}

GPUContextId::GPUContextId(GPUContext context) {
#if defined(KMM_USE_HIP)
    // HIP has no equivalent of `cuCtxGetId`. The context handle itself is
    // already a stable, unique identity, so it is used directly.
    m_id = reinterpret_cast<unsigned long long>(context);
#else
    KMM_GPU_CHECK(gpuCtxGetId(context, &m_id));
#endif
}

GPUContext context_from_stream(GPUStream stream) {
    GPUContext context;
#if defined(KMM_USE_HIP)
    // HIP has no equivalent of `cuStreamGetCtx`. Instead, resolve the device
    // the stream was created on and reuse its primary context, which is what
    // `SystemInfo` already treats as "the" context for that device. The
    // primary context is retained for the lifetime of the process (see
    // `system_info.cpp`), so this extra retain/release pair only needs the
    // handle value, not a lasting reference.
    GPUDevice device;
    KMM_GPU_CHECK(gpuStreamGetDevice(stream, &device));
    KMM_GPU_CHECK(gpuDevicePrimaryCtxRetain(&context, device));
    gpuDevicePrimaryCtxRelease(device);
#else
    KMM_GPU_CHECK(gpuStreamGetCtx(stream, &context));
#endif
    return context;
}

GPUStreamId::GPUStreamId(GPUStream stream) : GPUStreamId(stream, context_from_stream(stream)) {}

GPUStreamId::GPUStreamId(GPUStream stream, GPUContext context) : m_context_id(context) {
#if defined(KMM_USE_HIP)
    // HIP has no equivalent of `cuStreamGetId`. The stream handle itself is
    // already a stable, unique identity, so it is used directly.
    m_id = reinterpret_cast<unsigned long long>(stream);
#else
    KMM_GPU_CHECK(gpuStreamGetId(stream, &m_id));
#endif
}

GPUStreamId::GPUStreamId(const GPUStreamRef& stream) : GPUStreamId(stream.stream_id()) {}

GPUStreamId::GPUStreamId(const GPUStreamOwner& stream) :
    GPUStreamId(static_cast<GPUStreamRef>(stream)) {}

GPUStreamRef::GPUStreamRef(GPUStream stream) :
    m_context(context_from_stream(stream)),
    m_stream(stream),
    m_stream_id(stream, m_context) {}

GPUStreamOwner::GPUStreamOwner(GPUContext context, unsigned int flags) :
    m_stream([&]() {
        GPUStream result;
        GPUContextGuard guard(context);
        KMM_GPU_CHECK(gpuStreamCreate(&result, flags));
        return result;
    }()) {}

GPUStreamOwner::~GPUStreamOwner() {
    destroy();
}

void GPUStreamOwner::destroy() noexcept {
    if (m_stream != nullptr) {
        gpuStreamDestroy(m_stream);
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
