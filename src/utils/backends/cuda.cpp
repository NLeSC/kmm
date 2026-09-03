#include "kmm/runtime/memops/types.hpp"
#include "kmm/utils/backends.hpp"

namespace kmm {

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
    return cuMemcpyPeerAsync(
        dstDevicePtr,
        dstContext,
        srcDevicePtr,
        srcContext,
        ByteCount,
        hStream
    );
}

}  // namespace kmm