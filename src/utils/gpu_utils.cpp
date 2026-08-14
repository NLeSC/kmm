#include "fmt/format.h"

#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

void cuda_throw_exception(CUresult result, const char* file, int line, const char* expression) {
    const char* name = "UNKNOWN_ERROR";
    const char* description = "unknown error";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);

    throw CUDAException(
        fmt::format("CUDA error: {} ({}) at {}:{}: {}", name, description, file, line, expression)
    );
}

CUDAContextGuard::CUDAContextGuard(CUcontext context) : m_context(context) {
    KMM_CUDA_CHECK(cuCtxPushCurrent(context));
}

CUDAContextGuard::~CUDAContextGuard() {
    // Destructors must not throw, so the pop result is discarded rather than checked.
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
}

CUDAContextId::CUDAContextId(CUcontext context) {
    KMM_CUDA_CHECK(cuCtxGetId(context, &m_id));
}

CUcontext context_from_stream(CUstream stream) {
    CUcontext context;
    KMM_CUDA_CHECK(cuStreamGetCtx(stream, &context));
    return context;
}

CUDAStreamId::CUDAStreamId(CUstream stream) : CUDAStreamId(stream, context_from_stream(stream)) {}

CUDAStreamId::CUDAStreamId(CUstream stream, CUcontext context) : m_context_id(context) {
    KMM_CUDA_CHECK(cuStreamGetId(stream, &m_id));
}

CUDAStreamId::CUDAStreamId(const CUDAStreamRef& stream) : CUDAStreamId(stream.stream_id()) {}

CUDAStreamId::CUDAStreamId(const CUDAStream& stream) :
    CUDAStreamId(static_cast<CUDAStreamRef>(stream)) {}

CUDAStreamRef::CUDAStreamRef(CUstream stream) :
    m_context(context_from_stream(stream)),
    m_stream(stream),
    m_stream_id(stream, m_context) {}

CUDAStream::CUDAStream(CUcontext context, unsigned int flags) :
    m_stream([&]() {
        CUstream result;
        CUDAContextGuard guard(context);
        KMM_CUDA_CHECK(cuStreamCreate(&result, flags));
        return result;
    }()) {}

CUDAStream::~CUDAStream() {
    destroy();
}

void CUDAStream::destroy() noexcept {
    if (m_stream != nullptr) {
        cuStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

std::ostream& operator<<(std::ostream& stream, const CUDAStream& self) {
    if (self.m_stream == nullptr) {
        return stream << "CUDA-stream: none";
    }

    return stream << CUDAStreamRef(self.m_stream);
}

std::ostream& operator<<(std::ostream& stream, const CUDAStreamRef& self) {
    return stream << self.m_stream_id;
}

std::ostream& operator<<(std::ostream& stream, const CUDAStreamId& self) {
    return stream << "CUDA-stream:" << self.m_id;
}

std::ostream& operator<<(std::ostream& stream, const CUDAContextId& self) {
    return stream << "CUDA-context:" << self.m_id;
}

}  // namespace kmm
