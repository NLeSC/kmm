#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

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
