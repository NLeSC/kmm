#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

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
