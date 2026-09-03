#include "kmm/runtime/memops/types.hpp"
#include "kmm/utils/backends.hpp"

namespace kmm {

g_result_t g_stream_get_ctx(g_stream_t hStream, g_context_t* pctx) {
    // HIP has no equivalent of `cuStreamGetCtx`. Instead, resolve the device
    // the stream was created on and reuse its primary context, which is what
    // `SystemInfo` already treats as "the" context for that device.
    g_device_t device;

    g_stream_get_device(hStream, &device);
    g_device_primary_ctx_retain(pctx, device);
    g_device_primary_ctx_release(device);
}

const char* blas_get_status_name(blas_status_t) {
    return "";
}

g_result_t g_memcpy_async(
    g_device_ptr_t dst,
    g_device_ptr_t src,
    size_t ByteCount,
    g_stream_t hStream
) {
    return hipMemcpyAsync(dst, src, ByteCount, hipMemcpyDefault, hStream);
}

g_result_t g_memcpy_h_to_d_async(
    g_device_ptr_t dstDevice,
    const void* srcHost,
    size_t ByteCount,
    g_stream_t hStream
) {
    return hipMemcpyHtoDAsync(dstDevice, const_cast<void*>(srcHost), ByteCount, hStream);
}

g_result_t g_memcpy_h_to_d(g_device_ptr_t dstDevice, const void* srcHost, size_t ByteCount) {
    return hipMemcpyHtoD(dstDevice, const_cast<void*>(srcHost), ByteCount);
}

g_result_t g_memcpy_peer_async(
    g_device_ptr_t dstDevicePtr,
    g_context_t dstContext,
    g_device_t dstDevice,
    g_device_ptr_t srcDevicePtr,
    g_context_t srcContext,
    g_device_t srcDevice,
    size_t ByteCount,
    g_stream_t hStream
) {
    return hipMemcpyPeerAsync(dstDevicePtr, dstDevice, srcDevicePtr, srcDevice, ByteCount, hStream);
}

}  // namespace kmm