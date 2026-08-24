#pragma once

// Thin portability layer between the CUDA driver API and the HIP driver API.

#if defined(KMM_USE_CUDA)
    #include <cuda.h>
    // Only pulled in for the `dim3` vector type; the driver API itself lives entirely in <cuda.h>.
    #include <cuda_runtime.h>
#elif defined(KMM_USE_HIP)
    #include <hip/hip_runtime.h>
#else
    #error "gpu_api.hpp requires either KMM_USE_CUDA or KMM_USE_HIP to be defined"
#endif

namespace kmm {

#if defined(KMM_USE_CUDA)

using GPUResult = CUresult;
using GPUDevice = CUdevice;
using GPUContext = CUcontext;
using GPUStream = CUstream;
using GPUEvent = CUevent;
using GPUDeviceptr = CUdeviceptr;
using GPUMemPool = CUmemoryPool;
using GPUDeviceAttribute = CUdevice_attribute;

#elif defined(KMM_USE_HIP)

using GPUResult = hipError_t;
using GPUDevice = hipDevice_t;
using GPUContext = hipCtx_t;
using GPUStream = hipStream_t;
using GPUEvent = hipEvent_t;
using GPUDeviceptr = hipDeviceptr_t;
using GPUMemPool = hipMemPool_t;

#endif

}  // namespace kmm

// ---------------------------------------------------------------------------
// Result codes
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define GPU_SUCCESS             CUDA_SUCCESS
    #define GPU_ERROR_OUT_OF_MEMORY CUDA_ERROR_OUT_OF_MEMORY
    #define GPU_ERROR_NOT_READY     CUDA_ERROR_NOT_READY
#elif defined(KMM_USE_HIP)
    #define GPU_SUCCESS hipSuccess
#endif

// ---------------------------------------------------------------------------
// Flags / constants
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define GPU_STREAM_DEFAULT_FLAGS CU_STREAM_NON_BLOCKING

    #define GPU_DEVICE_ATTRIBUTE_MAX CU_DEVICE_ATTRIBUTE_MAX
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR \
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR \
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    #define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#elif defined(KMM_USE_HIP)
    #define GPU_STREAM_DEFAULT_FLAGS hipStreamNonBlocking

    #define GPU_DEVICE_ATTRIBUTE_MAX                      hipDeviceAttributeMax
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR hipDeviceAttributeComputeCapabilityMajor
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR hipDeviceAttributeComputeCapabilityMinor
    #define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    hipDeviceAttributeMaxThreadsPerBlock
#endif

// ---------------------------------------------------------------------------
// Init / error handling
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define gpuInit(flags)                 cuInit(flags)
    #define gpuGetErrorName(result, name)  cuGetErrorName(result, name)
    #define gpuGetErrorString(result, str) cuGetErrorString(result, str)
    #define gpuGetLastError()              cudaGetLastError()
    #define gpuGetErrorStringRuntime(err)  cudaGetErrorString(err)
#elif defined(KMM_USE_HIP)
    #define gpuInit(flags)                 hipInit(flags)
    #define gpuGetErrorName(result, name)  (*(name) = hipGetErrorName(result))
    #define gpuGetErrorString(result, str) (*(str) = hipGetErrorString(result))
    #define gpuGetLastError()              hipGetLastError()
    #define gpuGetErrorStringRuntime(err)  hipGetErrorString(err)
#endif

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define gpuDeviceGetCount(count)                     cuDeviceGetCount(count)
    #define gpuDeviceGet(device, ordinal)                cuDeviceGet(device, ordinal)
    #define gpuDeviceGetName(name, len, device)          cuDeviceGetName(name, len, device)
    #define gpuDeviceTotalMem(bytes, device)             cuDeviceTotalMem(bytes, device)
    #define gpuDeviceGetAttribute(value, attrib, device) cuDeviceGetAttribute(value, attrib, device)
    #define gpuDeviceCanAccessPeer(result, dev, peer)    cuDeviceCanAccessPeer(result, dev, peer)
    #define gpuDeviceGetDefaultMemPool(pool, device)     cuDeviceGetDefaultMemPool(pool, device)
    #define gpuDevicePrimaryCtxRetain(ctx, device)       cuDevicePrimaryCtxRetain(ctx, device)
    #define gpuDevicePrimaryCtxRelease(device)           cuDevicePrimaryCtxRelease(device)
#elif defined(KMM_USE_HIP)
    #define gpuDeviceGetCount(count)            hipGetDeviceCount(count)
    #define gpuDeviceGet(device, ordinal)       hipDeviceGet(device, ordinal)
    #define gpuDeviceGetName(name, len, device) hipDeviceGetName(name, len, device)
    #define gpuDeviceTotalMem(bytes, device)    hipDeviceTotalMem(bytes, device)
    #define gpuDeviceGetAttribute(value, attrib, device) \
        hipDeviceGetAttribute(value, attrib, device)
    #define gpuDeviceCanAccessPeer(result, dev, peer) hipDeviceCanAccessPeer(result, dev, peer)
    #define gpuDeviceGetDefaultMemPool(pool, device)  hipDeviceGetDefaultMemPool(pool, device)
    #define gpuDevicePrimaryCtxRetain(ctx, device)    hipDevicePrimaryCtxRetain(ctx, device)
    #define gpuDevicePrimaryCtxRelease(device)        hipDevicePrimaryCtxRelease(device)
#endif

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define gpuCtxGetDevice(device) cuCtxGetDevice(device)
    #define gpuCtxGetId(ctx, id)    cuCtxGetId(ctx, id)
    #define gpuCtxPushCurrent(ctx)  cuCtxPushCurrent(ctx)
    #define gpuCtxPopCurrent(ctx)   cuCtxPopCurrent(ctx)
#elif defined(KMM_USE_HIP)
    #define gpuCtxGetDevice(device) hipCtxGetDevice(device)
// HIP has no equivalent of cuCtxGetId; callers derive a context identity
// from the `GPUContext` handle itself instead (see `gpu_utils.cpp`).
    #define gpuCtxPushCurrent(ctx)  hipCtxPushCurrent(ctx)
    #define gpuCtxPopCurrent(ctx)   hipCtxPopCurrent(ctx)
#endif

inline kmm::GPUResult gpuCtxEnablePeerAccess(kmm::GPUContext peer_context) {
#if defined(KMM_USE_CUDA)
    return cuCtxEnablePeerAccess(peer_context, 0);
#elif defined(KMM_USE_HIP)
    return hipCtxEnablePeerAccess(peer_context, 0);
#endif
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define gpuStreamCreate(stream, flags) cuStreamCreate(stream, flags)
    #define gpuStreamDestroy(stream)       cuStreamDestroy(stream)
    #define gpuStreamGetCtx(stream, ctx)   cuStreamGetCtx(stream, ctx)
    #define gpuStreamGetId(stream, id)     cuStreamGetId(stream, id)
    #define gpuStreamSynchronize(stream)   cuStreamSynchronize(stream)
    #define gpuStreamQuery(stream)         cuStreamQuery(stream)
#elif defined(KMM_USE_HIP)
    #define gpuStreamCreate(stream, flags)     hipStreamCreateWithFlags(stream, flags)
    #define gpuStreamDestroy(stream)           hipStreamDestroy(stream)
    // HIP has no equivalent of cuStreamGetCtx/cuStreamGetId; `gpu_utils.cpp`
    // reconstructs the owning context via `gpuStreamGetDevice` plus the
    // device's (already-retained) primary context, and derives a stream
    // identity from the `GPUStream` handle itself.
    #define gpuStreamGetDevice(stream, device) hipStreamGetDevice(stream, device)
    #define gpuStreamSynchronize(stream)       hipStreamSynchronize(stream)
    #define gpuStreamQuery(stream)             hipStreamQuery(stream)
#endif

inline kmm::GPUResult gpuStreamWaitEvent(kmm::GPUStream stream, kmm::GPUEvent event) {
#if defined(KMM_USE_CUDA)
    return cuStreamWaitEvent(stream, event, 0);
#elif defined(KMM_USE_HIP)
    return hipStreamWaitEvent(stream, event, 0);
#endif
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define gpuEventDestroy(event)        cuEventDestroy(event)
    #define gpuEventRecord(event, stream) cuEventRecord(event, stream)
    #define gpuEventQuery(event)          cuEventQuery(event)
    #define gpuEventSynchronize(event)    cuEventSynchronize(event)
#elif defined(KMM_USE_HIP)
    #define gpuEventDestroy(event)        hipEventDestroy(event)
    #define gpuEventRecord(event, stream) hipEventRecord(event, stream)
    #define gpuEventQuery(event)          hipEventQuery(event)
    #define gpuEventSynchronize(event)    hipEventSynchronize(event)
#endif

inline kmm::GPUResult gpuEventCreate(kmm::GPUEvent* event) {
#if defined(KMM_USE_CUDA)
    return cuEventCreate(event, CU_EVENT_DISABLE_TIMING);
#elif defined(KMM_USE_HIP)
    return hipEventCreateWithFlags(event, hipEventDisableTiming);
#endif
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

#if defined(KMM_USE_CUDA)
    #define gpuMemAlloc(ptr, size) cuMemAlloc(ptr, size)
    #define gpuMemFree(ptr)        cuMemFree(ptr)
    #define gpuMemFreeHost(ptr)    cuMemFreeHost(ptr)

    #define gpuMemcpyAsync(dst, src, size, stream)     cuMemcpyAsync(dst, src, size, stream)
    #define gpuMemcpy2DAsync(params, stream)           cuMemcpy2DAsync(params, stream)
    #define gpuMemcpyHtoDAsync(dst, src, size, stream) cuMemcpyHtoDAsync(dst, src, size, stream)
    #define gpuMemcpyDtoHAsync(dst, src, size, stream) cuMemcpyDtoHAsync(dst, src, size, stream)
    #define gpuMemcpyPeerAsync(dst, dstCtx, src, srcCtx, size, stream) \
        cuMemcpyPeerAsync(dst, dstCtx, src, srcCtx, size, stream)

    #define gpuMemAllocAsync(ptr, size, stream) cuMemAllocAsync(ptr, size, stream)
    #define gpuMemFreeAsync(ptr, stream)        cuMemFreeAsync(ptr, stream)
    #define gpuMemAllocFromPoolAsync(ptr, size, pool, stream) \
        cuMemAllocFromPoolAsync(ptr, size, pool, stream)

    #define gpuMemPoolCreate(pool, props)          cuMemPoolCreate(pool, props)
    #define gpuMemPoolDestroy(pool)                cuMemPoolDestroy(pool)
    #define gpuMemPoolTrimTo(pool, minBytesToKeep) cuMemPoolTrimTo(pool, minBytesToKeep)

    #define gpuMemsetD8Async(ptr, value, count, stream)  cuMemsetD8Async(ptr, value, count, stream)
    #define gpuMemsetD16Async(ptr, value, count, stream) cuMemsetD16Async(ptr, value, count, stream)
    #define gpuMemsetD32Async(ptr, value, count, stream) cuMemsetD32Async(ptr, value, count, stream)
    #define gpuMemsetD2D8Async(ptr, pitch, value, width, height, stream) \
        cuMemsetD2D8Async(ptr, pitch, value, width, height, stream)
    #define gpuMemsetD2D16Async(ptr, pitch, value, width, height, stream) \
        cuMemsetD2D16Async(ptr, pitch, value, width, height, stream)
    #define gpuMemsetD2D32Async(ptr, pitch, value, width, height, stream) \
        cuMemsetD2D32Async(ptr, pitch, value, width, height, stream)
#elif defined(KMM_USE_HIP)
    #define gpuMemAlloc(ptr, size) hipMemAlloc(ptr, size)
    #define gpuMemFree(ptr)        hipMemFree(ptr)
    #define gpuMemFreeHost(ptr)    hipMemFreeHost(ptr)

    #define gpuMemcpyAsync(dst, src, size, stream)     hipMemcpyAsync(dst, src, size, stream)
    #define gpuMemcpy2DAsync(params, stream)           hipMemcpyParam2DAsync(params, stream)
    #define gpuMemcpyHtoDAsync(dst, src, size, stream) hipMemcpyHtoDAsync(dst, src, size, stream)
    #define gpuMemcpyDtoHAsync(dst, src, size, stream) hipMemcpyDtoHAsync(dst, src, size, stream)
    #define gpuMemcpyPeerAsync(dst, dstCtx, src, srcCtx, size, stream) \
        hipMemcpyPeerAsync(dst, dstCtx, src, srcCtx, size, stream)

    #define gpuMemAllocAsync(ptr, size, stream) hipMallocAsync(ptr, size, stream)
    #define gpuMemFreeAsync(ptr, stream)        hipFreeAsync(ptr, stream)
    #define gpuMemAllocFromPoolAsync(ptr, size, pool, stream) \
        hipMallocFromPoolAsync(ptr, size, pool, stream)

    #define gpuMemPoolCreate(pool, props)          hipMemPoolCreate(pool, props)
    #define gpuMemPoolDestroy(pool)                hipMemPoolDestroy(pool)
    #define gpuMemPoolTrimTo(pool, minBytesToKeep) hipMemPoolTrimTo(pool, minBytesToKeep)

    #define gpuMemsetD8Async(ptr, value, count, stream) hipMemsetD8Async(ptr, value, count, stream)
    #define gpuMemsetD16Async(ptr, value, count, stream) \
        hipMemsetD16Async(ptr, value, count, stream)
    #define gpuMemsetD32Async(ptr, value, count, stream) \
        hipMemsetD32Async(ptr, value, count, stream)
    #define gpuMemsetD2D8Async(ptr, pitch, value, width, height, stream) \
        hipMemsetD2D8Async(ptr, pitch, value, width, height, stream)
    #define gpuMemsetD2D16Async(ptr, pitch, value, width, height, stream) \
        hipMemsetD2D16Async(ptr, pitch, value, width, height, stream)
    #define gpuMemsetD2D32Async(ptr, pitch, value, width, height, stream) \
        hipMemsetD2D32Async(ptr, pitch, value, width, height, stream)
#endif

// gpuMemAllocManaged always requests global attach at every call site in
// this codebase, so the flag is baked in here instead of passed by callers.
inline kmm::GPUResult gpuMemAllocManaged(kmm::GPUDeviceptr* ptr, size_t size) {
#if defined(KMM_USE_CUDA)
    return cuMemAllocManaged(ptr, size, CU_MEM_ATTACH_GLOBAL);
#elif defined(KMM_USE_HIP)
    return hipMallocManaged(reinterpret_cast<void**>(ptr), size, hipMemAttachGlobal);
#endif
}

// gpuMemHostAlloc always requests portable, device-mapped host memory at
// every call site in this codebase, so the flags are baked in here instead
// of passed by callers.
inline kmm::GPUResult gpuMemHostAlloc(void** ptr, size_t size) {
#if defined(KMM_USE_CUDA)
    return cuMemHostAlloc(ptr, size, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
#elif defined(KMM_USE_HIP)
    return hipHostMalloc(ptr, size, hipHostMallocPortable | hipHostMallocMapped);
#endif
}
