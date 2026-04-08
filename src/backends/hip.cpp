#ifdef KMM_USE_HIP

#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

const char* blasGetStatusName(blasStatus_t) {
    return "";
}

GPUresult gpuMemcpyAsync(
    GPUdeviceptr dst,
    GPUdeviceptr src,
    size_t ByteCount,
    GPUstream_t hStream
) {
    return hipMemcpyAsync(dst, src, ByteCount, hipMemcpyDefault, hStream);
}

GPUresult gpuMemcpyHtoDAsync(
    GPUdeviceptr dstDevice,
    const void* srcHost,
    size_t ByteCount,
    GPUstream_t hStream
) {
    return hipMemcpyHtoDAsync(dstDevice, const_cast<void*>(srcHost), ByteCount, hStream);
}

GPUresult gpuMemcpyHtoD(GPUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    return hipMemcpyHtoD(dstDevice, const_cast<void*>(srcHost), ByteCount);
}

GPUresult gpuMemcpyPeerAsync(
    GPUdeviceptr dstDevicePtr,
    GPUcontext dstContext,
    GPUdevice dstDevice,
    GPUdeviceptr srcDevicePtr,
    GPUcontext srcContext,
    GPUdevice srcDevice,
    size_t ByteCount,
    GPUstream_t hStream
) {
    return hipMemcpyPeerAsync(dstDevicePtr, dstDevice, srcDevicePtr, srcDevice, ByteCount, hStream);
}

}  // namespace kmm

#endif  // KMM_USE_HIP
