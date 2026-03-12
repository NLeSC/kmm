
#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

#if !defined(KMM_USE_CUDA) && !defined(KMM_USE_HIP)

GPUresult gpuCtxGetDevice(GPUdevice* device) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGetName(char* name, int len, GPUdevice dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGetAttribute(int* value, GPUdevice_attribute attribute, GPUdevice dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemGetInfo(size_t* free, size_t* total) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyHtoDAsync(GPUdeviceptr dev, const void* host, size_t size, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoHAsync(void* host, GPUdeviceptr device, size_t size, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyPeerAsync(
    GPUdeviceptr dst_ptr,
    GPUcontext dst_ctx,
    GPUdevice dst_id,
    GPUdeviceptr std_ptr,
    GPUcontext src_ctx,
    GPUdevice src_id,
    size_t size,
    GPUstream_t stream
) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamSynchronize(GPUstream_t dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD8Async(GPUdeviceptr dev, unsigned char val, size_t size, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD16Async(GPUdeviceptr dev, unsigned short val, size_t size, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD32Async(GPUdeviceptr dev, unsigned int val, size_t size, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

gpuError_t gpuMemsetAsync(void* dst, int value, size_t sizeBytes, GPUstream_t stream) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, enum gpuMemcpyKind kind) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyAsync(GPUdeviceptr dst, GPUdeviceptr src, size_t size, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemHostAlloc(void** host, size_t size, unsigned int flags) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemFreeHost(void* dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemAlloc(GPUdeviceptr* dev, size_t size) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemAllocAsync(GPUdeviceptr*, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

gpuError_t gpuMalloc(GPUdeviceptr*, size_t) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemFree(GPUdeviceptr dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

gpuError_t gpuFree(GPUdeviceptr) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemPoolCreate(GPUmemoryPool* pool, const GPUmemPoolProps* props) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemPoolDestroy(GPUmemoryPool pool) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemAllocFromPoolAsync(
    GPUdeviceptr* dev,
    size_t size,
    GPUmemoryPool pool,
    GPUstream_t stream
) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemFreeAsync(GPUdeviceptr dev, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

gpuError_t gpuFreeAsync(GPUdeviceptr, GPUstream_t) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuGetStreamPriorityRange(int* least, int* greatest) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamCreateWithPriority(GPUstream_t* stream, unsigned int flags, int priority) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamQuery(GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamDestroy(GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventSynchronize(GPUevent_t event) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventRecord(GPUevent_t event, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamWaitEvent(GPUstream_t stream, GPUevent_t event, unsigned int flags) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventQuery(GPUevent_t event) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventDestroy(GPUevent_t event) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventCreate(GPUevent_t event, unsigned int flags) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyHtoD(GPUdeviceptr dest, const void* src, size_t size) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpy2DAsync(const GPU_MEMCPY2D* dev, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpy2D(const GPU_MEMCPY2D* dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoH(void* dest, GPUdeviceptr src, size_t size) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoDAsync(GPUdeviceptr dest, GPUdeviceptr src, size_t size, GPUstream_t stream) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoD(GPUdeviceptr dest, GPUdeviceptr src, size_t size) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuGetErrorName(GPUresult error, const char** name) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuGetErrorString(GPUresult error, const char** desc) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

const char* GPUrtGetErrorName(gpuError_t error) {
    return "";
}

const char* GPUrtGetErrorString(gpuError_t error) {
    return "";
}

gpuError_t gpuGetLastError(void) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuInit(unsigned int flags) {
    return GPUresult(GPU_ERROR_NO_DEVICE);
}

GPUresult gpuDeviceGetCount(int* count) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGet(GPUdevice* dev, int ordinal) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuPointerGetAttribute(void* prt, GPUpointer_attribute attribute, GPUdeviceptr dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxCreate(GPUcontext* ctx, unsigned int flags, GPUdevice dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxDestroy(GPUcontext ctx) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDevicePrimaryCtxRetain(GPUcontext* ctx, GPUdevice dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDevicePrimaryCtxRelease(GPUdevice dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxPushCurrent(GPUcontext ctx) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxPopCurrent(GPUcontext* ctx) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}
gpuError_t GPUrtLaunchKernel(
    const void* func,
    dim3 grid,
    dim3 block,
    void** args,
    size_t smem,
    GPUstream_t stream
) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemPoolTrimTo(GPUmemoryPool pool, size_t size) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGetDefaultMemPool(GPUmemoryPool* pool, GPUdevice dev) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemPoolSetAttribute(GPUmemoryPool memPool, gpuMemPoolAttr attr, void* value) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

gpuError_t gpuDeviceSynchronize() {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

gpuError_t gpuEventElapsedTime(float* ms, GPUevent_t start, GPUevent_t stop) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

blasStatus_t blasCreate(blasHandle_t blas) {
    return blasStatus_t(1);
}

blasStatus_t blasSetStream(blasHandle_t blas, GPUstream_t stream) {
    return blasStatus_t(1);
}

blasStatus_t blasDestroy(blasHandle_t blas) {
    return blasStatus_t(1);
}

const char* blasGetStatusName(blasStatus_t blas) {
    return "";
}

const char* blasGetStatusString(blasStatus_t blas) {
    return "";
}

void execute_gpu_fill_async(
    GPUstream_t stream,
    GPUdeviceptr dst_buffer,
    size_t nbytes,
    const void* pattern,
    size_t pattern_nbytes
) {}

void execute_gpu_reduction_async(
    GPUstream_t stream,
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    ReductionDef reduction
) {}

void execute_gpu_fill_async(GPUstream_t stream, GPUdeviceptr dst_buffer, const FillDef& fill) {}

#elif defined(KMM_USE_HIP)

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

#elif defined(KMM_USE_CUDA)

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

#endif

}  // namespace kmm