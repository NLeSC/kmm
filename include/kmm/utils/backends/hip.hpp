#pragma once

#include <cstddef>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "kmm/core/macros.hpp"

namespace kmm {

// Types

using half_type = __half;
using bfloat16_type = __hip_bfloat16;
using g_result_t = hipError_t;
using gpu_error_t = hipError_t;
using g_device_t = hipDevice_t;
using g_device_attribute_t = hipDeviceAttribute_t;
using g_context_t = hipCtx_t;
using g_memory_type_t = hipMemoryType;
using g_stream_t = hipStream_t;
using g_device_ptr_t = hipDeviceptr_t;
using g_memory_pool_t = hipMemPool_t;
using g_mem_pool_props_t = hipMemPoolProps;
using g_mem_allocation_type_t = hipMemAllocationType;
using g_mem_allocation_handle_type_t = hipMemAllocationHandleType;
using g_mem_location_type_t = hipMemLocationType;
using gpu_event_t = hipEvent_t;
using g_event_t = hipEvent_t;
using gpu_memcpy2d_t = hip_Memcpy2D;
using gpu_mem_pool_t = hipMemPool_t;
using gpu_mem_pool_attr_t = hipMemPoolAttr;

// Device Management Constants & Functions

#define G_DEVICE_ATTRIBUTE_MAX                      0
#define G_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    hipDeviceAttributeMaxThreadsPerBlock
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          hipDeviceAttributeMaxBlockDimX
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          hipDeviceAttributeMaxBlockDimY
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          hipDeviceAttributeMaxBlockDimZ
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           hipDeviceAttributeMaxGridDimX
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           hipDeviceAttributeMaxGridDimY
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           hipDeviceAttributeMaxGridDimZ
#define G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR hipDeviceAttributeComputeCapabilityMajor
#define G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR hipDeviceAttributeComputeCapabilityMinor
#define GPU_ERROR_PEER_ACCESS_ALREADY_ENABLED       hipErrorPeerAccessAlreadyEnabled

#define g_init                       hipInit
#define g_device_get_count           hipGetDeviceCount
#define g_device_get                 hipDeviceGet
#define g_device_get_name            hipDeviceGetName
#define g_device_total_mem           hipDeviceTotalMem
#define g_device_get_attribute       hipDeviceGetAttribute
#define g_device_can_access_peer     hipDeviceCanAccessPeer
#define g_ctx_get_device             hipCtxGetDevice
#define g_ctx_create                 hipCtxCreate
#define g_ctx_destroy                hipCtxDestroy
#define g_device_primary_ctx_retain  hipDevicePrimaryCtxRetain
#define g_device_primary_ctx_release hipDevicePrimaryCtxRelease
#define g_ctx_push_current           hipCtxPushCurrent
#define g_ctx_pop_current            hipCtxPopCurrent
#define g_ctx_enable_peer_access     hipCtxEnablePeerAccess

// Stream Management Constants & Functions

#define G_STREAM_NON_BLOCKING           hipStreamNonBlocking
#define g_ctx_get_stream_priority_range hipDeviceGetStreamPriorityRange
#define g_stream_create                 hipStreamCreateWithFlags
#define g_stream_create_with_priority   hipStreamCreateWithPriority
#define g_stream_get_device             hipStreamGetDevice
#define g_stream_get_id                 hipStreamGetId
#define g_stream_query                  hipStreamQuery
#define g_stream_synchronize            hipStreamSynchronize
#define g_stream_destroy                hipStreamDestroy
#define g_stream_wait_event             hipStreamWaitEvent

static inline g_result_t g_stream_get_ctx(g_stream_t hStream, g_context_t* pctx) {
    // HIP has no equivalent of `cuStreamGetCtx`. Instead, resolve the device
    // the stream was created on and reuse its primary context, which is what
    // `SystemInfo` already treats as "the" context for that device.
    g_device_t device;

    KMM_GPU_CHECK(g_stream_get_device(hStream, &device));
    KMM_GPU_CHECK(g_device_primary_ctx_retain(pctx, device));
    KMM_GPU_CHECK(g_device_primary_ctx_release(device));
}

// Event Management Constants & Functions

#define G_EVENT_WAIT_DEFAULT   0
#define G_EVENT_DISABLE_TIMING hipEventDisableTiming

#define gpu_event_create       hipEventCreate
#define g_event_create         hipEventCreateWithFlags
#define gpu_event_record       hipEventRecord
#define g_event_record         hipEventRecord
#define g_event_query          hipEventQuery
#define gpu_event_synchronize  hipEventSynchronize
#define g_event_synchronize    hipEventSynchronize
#define gpu_event_destroy      hipEventDestroy
#define g_event_destroy        hipEventDestroy
#define gpu_event_elapsed_time hipEventElapsedTime

// Memory Management Constants & Functions

#define G_MEMHOSTALLOC_PORTABLE            hipHostMallocPortable
#define G_MEMHOSTALLOC_DEVICEMAP           hipHostMallocMapped
#define G_MEMORYTYPE_HOST                  hipMemoryTypeHost
#define G_MEMORYTYPE_DEVICE                hipMemoryTypeDevice
#define G_POINTER_ATTRIBUTE_MEMORY_TYPE    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
#define G_POINTER_ATTRIBUTE_DEVICE_ORDINAL HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define G_MEM_ATTACH_GLOBAL                hipMemAttachGlobal

#define g_mem_get_info                hipMemGetInfo
#define gpu_mem_get_info              hipMemGetInfo
#define g_mem_alloc                   hipMalloc
#define g_mem_free                    hipFree
#define gpu_malloc                    hipMalloc
#define gpu_free                      hipFree
#define g_mem_host_alloc              hipHostMalloc
#define g_mem_free_host               hipHostFree
#define g_pointer_get_attribute       hipPointerGetAttribute
#define g_mem_alloc_managed           hipMemAllocManaged
#define g_mem_host_get_device_pointer hipMemHostGetDevicePointer
#define g_mem_prefetch_async          hipMemPrefetchAsync

// Memory Copy Operations

#define gpu_memcpy                  hipMemcpy
#define gpu_memcpy_host_to_host     hipMemcpyHostToHost
#define gpu_memcpy_host_to_device   hipMemcpyHostToDevice
#define gpu_memcpy_device_to_host   hipMemcpyDeviceToHost
#define gpu_memcpy_device_to_device hipMemcpyDeviceToDevice
#define gpu_memcpy_default          hipMemcpyDefault

#define g_memcpy_d_to_h       hipMemcpyDtoH
#define g_memcpy_d_to_h_async hipMemcpyDtoHAsync
#define g_memcpy_d_to_d       hipMemcpyDtoD
#define g_memcpy_d_to_d_async hipMemcpyDtoDAsync

static inline g_result_t g_memcpy_async(
    g_device_ptr_t dst,
    g_device_ptr_t src,
    size_t ByteCount,
    g_stream_t hStream
) {
    return hipMemcpyAsync(dst, src, ByteCount, hipMemcpyDefault, hStream);
}

static inline g_result_t g_memcpy_h_to_d_async(
    g_device_ptr_t dstDevice,
    const void* srcHost,
    size_t ByteCount,
    g_stream_t hStream
) {
    return hipMemcpyHtoDAsync(dstDevice, const_cast<void*>(srcHost), ByteCount, hStream);
}

static inline g_result_t g_memcpy_h_to_d(
    g_device_ptr_t dstDevice,
    const void* srcHost,
    size_t ByteCount
) {
    return hipMemcpyHtoD(dstDevice, const_cast<void*>(srcHost), ByteCount);
}

static inline g_result_t g_memcpy_peer_async(
    g_device_ptr_t dstAddr,
    g_context_t dstContext,
    g_device_t dstDevice,
    g_device_ptr_t srcAddr,
    g_context_t srcContext,
    g_device_t srcDevice,
    size_t ByteCount,
    g_stream_t hStream
) {
    return hipMemcpyPeerAsync(dstAddr, dstDevice, srcAddr, srcDevice, ByteCount, hStream);
}

// Memory Fill Operations

#define g_memset_d8_async  hipMemsetD8Async
#define g_memset_d16_async hipMemsetD16Async
#define g_memset_d32_async hipMemsetD32Async
#define gpu_memset_async   hipMemsetAsync

// 2D Memory Operations

#define g_memcpy_2d       hipMemcpyParam2D
#define g_memcpy_2d_async hipMemcpyParam2DAsync

#define g_memset_2d_d8_async  hipMemsetD2D8Async
#define g_memset_2d_d16_async hipMemsetD2D16Async
#define g_memset_2d_d32_async hipMemsetD2D32Async

// Memory Pool Management Constants & Functions

#define G_MEM_ALLOCATION_TYPE_PINNED      hipMemAllocationTypePinned
#define G_MEM_HANDLE_TYPE_NONE            hipMemHandleTypeNone
#define G_MEM_LOCATION_TYPE_DEVICE        hipMemLocationTypeDevice
#define G_CTX_MAP_HOST                    0
#define g_mem_pool_attr_release_threshold hipMemPoolAttrReleaseThreshold

#define gpu_malloc_async                hipMallocAsync
#define g_mem_pool_create               hipMemPoolCreate
#define g_mem_pool_destroy              hipMemPoolDestroy
#define g_mem_alloc_from_pool_async     hipMallocFromPoolAsync
#define g_mem_alloc_async               hipMallocAsync
#define g_mem_free_async                hipFreeAsync
#define gpu_free_async                  hipFreeAsync
#define g_mem_pool_trim_to              hipMemPoolTrimTo
#define gpu_device_get_default_mem_pool hipDeviceGetDefaultMemPool
#define g_device_get_default_mem_pool   hipDeviceGetDefaultMemPool
#define gpu_mem_pool_set_attribute      hipMemPoolSetAttribute
#define g_mem_pool_set_attribute        hipMemPoolSetAttribute

// Error Handling Constants & Functions

// HIP has no separate driver/runtime API split, so G_* and GPU_* map to the same symbols.
#define G_SUCCESS             hipSuccess
#define G_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory
#define G_ERROR_UNKNOWN       hipErrorUnknown
#define G_ERROR_NOT_READY     hipErrorNotReady
#define G_ERROR_NO_DEVICE     hipErrorNoDevice

#define GPU_SUCCESS             hipSuccess
#define GPU_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory
#define GPU_ERROR_UNKNOWN       hipErrorUnknown
#define GPU_ERROR_NOT_READY     hipErrorNotReady
#define GPU_ERROR_NO_DEVICE     hipErrorNoDevice

#define g_get_error_name     hipDrvGetErrorName
#define g_get_error_string   hipDrvGetErrorString
#define gpu_get_error_name   hipGetErrorName
#define gpu_get_error_string hipGetErrorString
#define gpu_get_last_error   hipGetLastError

// Kernel Execution

#define gpu_launch_kernel      hipLaunchKernel
#define gpu_device_synchronize hipDeviceSynchronize

// BLAS Support

using blas_status_t = rocblas_status;
using blas_handle_t = rocblas_handle;

#define blas_create            rocblas_create_handle
#define blas_set_stream        rocblas_set_stream
#define blas_destroy           rocblas_destroy_handle
#define blas_get_status_string rocblas_status_to_string

static inline const char* blas_get_status_name(blas_status_t) {
    return "";
}

}  // namespace kmm
