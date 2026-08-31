#pragma once

#include <cstddef>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "kmm/core/macros.hpp"

namespace kmm {

// Types

using half_type = __half;
using bfloat16_type = __nv_bfloat16;
using g_result_t = CUresult;
using gpu_error_t = cudaError_t;
using g_device_t = CUdevice;
using g_device_attribute_t = CUdevice_attribute;
using g_context_t = CUcontext;
using g_memory_type_t = CUmemorytype;
using g_stream_t = CUstream;
using g_device_ptr_t = CUdeviceptr;
using g_memory_pool_t = CUmemoryPool;
using g_mem_pool_props_t = CUmemPoolProps;
using g_mem_allocation_type_t = CUmemAllocationType;
using g_mem_allocation_handle_type_t = CUmemAllocationHandleType;
using g_mem_location_type_t = CUmemLocationType;
using gpu_event_t = cudaEvent_t;
using g_event_t = CUevent;
using gpu_memcpy2d_t = CUDA_MEMCPY2D;
using gpu_mem_pool_t = cudaMemPool_t;
using gpu_mem_pool_attr_t = cudaMemPoolAttr;

// Device Management Constants & Functions

#define G_DEVICE_ATTRIBUTE_MAX                      CU_DEVICE_ATTRIBUTE_MAX
#define G_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
#define G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

#define g_init                       cuInit
#define g_device_get_count           cuDeviceGetCount
#define g_device_get                 cuDeviceGet
#define g_device_get_name            cuDeviceGetName
#define g_device_total_mem           cuDeviceTotalMem
#define g_device_get_attribute       cuDeviceGetAttribute
#define g_ctx_get_device             cuCtxGetDevice
#define g_ctx_create                 cuCtxCreate
#define g_ctx_destroy                cuCtxDestroy
#define g_device_primary_ctx_retain  cuDevicePrimaryCtxRetain
#define g_device_primary_ctx_release cuDevicePrimaryCtxRelease
#define g_ctx_push_current           cuCtxPushCurrent
#define g_ctx_pop_current            cuCtxPopCurrent

// Stream Management Constants & Functions

#define G_STREAM_NON_BLOCKING           CU_STREAM_NON_BLOCKING
#define g_ctx_get_stream_priority_range cuCtxGetStreamPriorityRange
#define g_stream_create_with_priority   cuStreamCreateWithPriority
#define g_stream_get_ctx                cuStreamGetCtx
#define g_stream_create                 cuStreamCreate
#define g_stream_query                  cuStreamQuery
#define g_stream_synchronize            cuStreamSynchronize
#define g_stream_destroy                cuStreamDestroy
#define g_stream_wait_event             cuStreamWaitEvent

// Event Management Constants & Functions

#define G_EVENT_WAIT_DEFAULT   CU_EVENT_WAIT_DEFAULT
#define G_EVENT_DISABLE_TIMING CU_EVENT_DISABLE_TIMING

#define gpu_event_create       cudaEventCreate
#define g_event_create         cuEventCreate
#define gpu_event_record       cudaEventRecord
#define g_event_record         cuEventRecord
#define g_event_query          cuEventQuery
#define gpu_event_synchronize  cudaEventSynchronize
#define g_event_synchronize    cuEventSynchronize
#define gpu_event_destroy      cudaEventDestroy
#define g_event_destroy        cuEventDestroy
#define gpu_event_elapsed_time cudaEventElapsedTime

// Memory Management Constants & Functions

#define G_MEMHOSTALLOC_PORTABLE            CU_MEMHOSTALLOC_PORTABLE
#define G_MEMHOSTALLOC_DEVICEMAP           CU_MEMHOSTALLOC_DEVICEMAP
#define G_MEMORYTYPE_HOST                  CU_MEMORYTYPE_HOST
#define G_MEMORYTYPE_DEVICE                CU_MEMORYTYPE_DEVICE
#define G_POINTER_ATTRIBUTE_MEMORY_TYPE    CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define G_POINTER_ATTRIBUTE_DEVICE_ORDINAL CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define G_MEM_ATTACH_GLOBAL                CU_MEM_ATTACH_GLOBAL

#define g_mem_get_info                cuMemGetInfo
#define gpu_mem_get_info              gpuMemGetInfo
#define g_mem_alloc                   cuMemAlloc
#define g_mem_free                    cuMemFree
#define gpu_malloc                    cudaMalloc
#define gpu_free                      cudaFree
#define g_mem_host_alloc              cuMemHostAlloc
#define g_mem_free_host               cuMemFreeHost
#define g_mem_alloc_manged            cuMemAllocManaged
#define g_mem_host_get_device_pointer cuMemHostGetDevicePointer
#define g_mem_prefetch_async          cuMemPrefetchAsync
#define g_pointer_get_attribute       cuPointerGetAttribute
#define g_device_can_access_peer      cuDeviceCanAccessPeer
#define g_ctx_enable_peer_access      cuCtxEnablePeerAccess

// Memory Copy Operations

#define gpu_memcpy                  cudaMemcpy
#define gpu_memcpy_host_to_host     cudaMemcpyHostToHost
#define gpu_memcpy_host_to_device   cudaMemcpyHostToDevice
#define gpu_memcpy_device_to_host   cudaMemcpyDeviceToHost
#define gpu_memcpy_device_to_device cudaMemcpyDeviceToDevice
#define gpu_memcpy_default          cudaMemcpyDefault

#define g_memcpy_h_to_d       cuMemcpyHtoD
#define g_memcpy_h_to_d_async cuMemcpyHtoDAsync
#define g_memcpy_d_to_h       cuMemcpyDtoH
#define g_memcpy_d_to_h_async cuMemcpyDtoHAsync
#define g_memcpy_d_to_d       cuMemcpyDtoD
#define g_memcpy_d_to_d_async cuMemcpyDtoDAsync
#define g_memcpy_async        cuMemcpyAsync

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
    return cuMemcpyPeerAsync(dstAddr, dstContext, srcAddr, srcContext, ByteCount, hStream);
}

// Memory Fill Operations

#define g_memset_d8_async  cuMemsetD8Async
#define g_memset_d16_async cuMemsetD16Async
#define g_memset_d32_async cuMemsetD32Async
#define gpu_memset_async   cudaMemsetAsync

// 2D Memory Operations

#define g_memcpy_2d       cuMemcpy2D
#define g_memcpy_2d_async cuMemcpy2DAsync

#define g_memset_2d_d8_async  cuMemsetD2D8Async
#define g_memset_2d_d16_async cuMemsetD2D16Async
#define g_memset_2d_d32_async cuMemsetD2D32Async

// Memory Pool Management Constants & Functions

#define G_MEM_ALLOCATION_TYPE_PINNED      CU_MEM_ALLOCATION_TYPE_PINNED
#define G_MEM_HANDLE_TYPE_NONE            CU_MEM_HANDLE_TYPE_NONE
#define G_MEM_LOCATION_TYPE_DEVICE        CU_MEM_LOCATION_TYPE_DEVICE
#define G_CTX_MAP_HOST                    CU_CTX_MAP_HOST
#define g_mem_pool_attr_release_threshold cuMemPoolAttrReleaseThreshold

#define gpu_malloc_async                cudaMallocAsync
#define g_mem_pool_create               cuMemPoolCreate
#define g_mem_pool_destroy              cuMemPoolDestroy
#define g_mem_alloc_from_pool_async     cuMemAllocFromPoolAsync
#define g_mem_alloc_async               cuMemAllocAsync
#define g_mem_free_async                cuMemFreeAsync
#define gpu_free_async                  cudaFreeAsync
#define g_mem_pool_trim_to              cuMemPoolTrimTo
#define gpu_device_get_default_mem_pool cudaDeviceGetDefaultMemPool
#define g_device_get_default_mem_pool   cuDeviceGetDefaultMemPool
#define gpu_mem_pool_set_attribute      cudaMemPoolSetAttribute
#define g_mem_pool_set_attribute        cuMemPoolSetAttribute

// Error Handling Constants & Functions

#define G_SUCCESS             CUDA_SUCCESS
#define G_ERROR_OUT_OF_MEMORY CUDA_ERROR_OUT_OF_MEMORY
#define G_ERROR_UNKNOWN       CUDA_ERROR_UNKNOWN
#define G_ERROR_NOT_READY     CUDA_ERROR_NOT_READY
#define G_ERROR_NO_DEVICE     CUDA_ERROR_NO_DEVICE

#define GPU_SUCCESS             cudaSuccess
#define GPU_ERROR_OUT_OF_MEMORY cudaErrorMemoryAllocation
#define GPU_ERROR_UNKNOWN       cudaErrorUnknown
#define GPU_ERROR_NOT_READY     cudaErrorNotReady
#define GPU_ERROR_NO_DEVICE     cudaErrorNoDevice

#define g_get_error_name     cuGetErrorName
#define g_get_error_string   cuGetErrorString
#define gpu_get_error_name   cudaGetErrorName
#define gpu_get_error_string cudaGetErrorString
#define gpu_get_last_error   cudaGetLastError

// Kernel Execution

#define gpu_launch_kernel      cudaLaunchKernel
#define gpu_device_synchronize cudaDeviceSynchronize

// BLAS Support

using blas_status_t = cublasStatus_t;
using blas_handle_t = cublasHandle_t;

#define blas_create            cublasCreate
#define blas_set_stream        cublasSetStream
#define blas_destroy           cublasDestroy
#define blas_get_status_name   cublasGetStatusName
#define blas_get_status_string cublasGetStatusString

}  // namespace kmm
