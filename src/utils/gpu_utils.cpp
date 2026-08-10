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

CUDAStream::CUDAStream(CUcontext context, unsigned int flags) {
    CUDAContextGuard guard(context);
    KMM_CUDA_CHECK(cuStreamCreate(&m_stream, flags));
}

CUDAStream::~CUDAStream() {
    if (m_stream != nullptr) {
        cuStreamDestroy(m_stream);
    }
}

cuda_context_id::cuda_context_id(CUcontext context) {
    KMM_CUDA_CHECK(cuCtxGetId(context, &m_id));
}

cuda_stream_id::cuda_stream_id(CUstream stream) :
    m_context_id([&] {
        CUcontext context;
        KMM_CUDA_CHECK(cuStreamGetCtx(stream, &context));
        return context;
    }()) {
    KMM_CUDA_CHECK(cuStreamGetId(stream, &m_id));
}

}  // namespace kmm
