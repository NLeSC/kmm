#pragma once

#include <cstddef>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "kmm/utils/macros.hpp"

namespace kmm {

// Types
using half_type = __half;
using bfloat16_type = __hip_bfloat16;
using GPUresult = hipError_t;
using gpuError_t = hipError_t;
using GPUdevice = hipDevice_t;
using GPUdevice_attribute = hipDeviceAttribute_t;
using GPUcontext = hipCtx_t;
using GPUmemorytype = hipMemoryType;
using GPUstream_t = hipStream_t;
using GPUdeviceptr = hipDeviceptr_t;
using GPUmemoryPool = hipMemPool_t;
using GPUmemPoolProps = hipMemPoolProps;
using GPUmemAllocationType = hipMemAllocationType;
using GPUmemAllocationHandleType = hipMemAllocationHandleType;
using GPUmemLocationType = hipMemLocationType;
using GPUevent_t = hipEvent_t;
using GPU_MEMCPY2D = hip_Memcpy2D;
using gpuMemPoolAttr = hipMemPoolAttr;

// Constants
#define GPU_DEVICE_ATTRIBUTE_MAX                      0
#define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    hipDeviceAttributeMaxThreadsPerBlock
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          hipDeviceAttributeMaxBlockDimX
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          hipDeviceAttributeMaxBlockDimY
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          hipDeviceAttributeMaxBlockDimZ
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           hipDeviceAttributeMaxGridDimX
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           hipDeviceAttributeMaxGridDimY
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           hipDeviceAttributeMaxGridDimZ
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR hipDeviceAttributeComputeCapabilityMajor
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR hipDeviceAttributeComputeCapabilityMinor
#define GPU_MEMHOSTALLOC_PORTABLE                     hipHostMallocPortable
#define GPU_MEMHOSTALLOC_DEVICEMAP                    hipHostMallocMapped
#define GPU_SUCCESS                                   hipSuccess
#define GPU_ERROR_OUT_OF_MEMORY                       hipErrorOutOfMemory
#define GPU_MEM_ALLOCATION_TYPE_PINNED                hipMemAllocationTypePinned
#define GPU_MEM_HANDLE_TYPE_NONE                      hipMemHandleTypeNone
#define GPU_MEM_LOCATION_TYPE_DEVICE                  hipMemLocationTypeDevice
#define GPU_ERROR_UNKNOWN                             hipErrorUnknown
#define GPU_STREAM_NON_BLOCKING                       hipStreamNonBlocking
#define GPU_EVENT_WAIT_DEFAULT                        0
#define GPU_ERROR_NOT_READY                           hipErrorNotReady
#define GPU_EVENT_DISABLE_TIMING                      hipEventDisableTiming
#define GPU_MEMORYTYPE_HOST                           hipMemoryTypeHost
#define GPU_MEMORYTYPE_DEVICE                         hipMemoryTypeDevice
#define GPU_ERROR_NO_DEVICE                           hipErrorNoDevice
#define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL          HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define GPU_CTX_MAP_HOST                              0

// HIP Functions
#define gpuCtxGetDevice                   hipCtxGetDevice
#define gpuDeviceGetName                  hipDeviceGetName
#define gpuDeviceGetAttribute             hipDeviceGetAttribute
#define gpuMemGetInfo                     hipMemGetInfo
#define gpuMemcpyDtoHAsync                hipMemcpyDtoHAsync
#define gpuStreamSynchronize              hipStreamSynchronize
#define gpuMemsetD8Async                  hipMemsetD8Async
#define gpuMemsetD16Async                 hipMemsetD16Async
#define gpuMemsetD32Async                 hipMemsetD32Async
#define gpuMemsetAsync                    hipMemsetAsync
#define gpuMemHostAlloc                   hipHostMalloc
#define gpuMemFreeHost                    hipHostFree
#define gpuMemAlloc                       hipMalloc
#define gpuMemAllocAsync                  hipMallocAsync
#define gpuMalloc                         hipMalloc
#define gpuMemFree                        hipFree
#define gpuFree                           hipFree
#define gpuMemPoolCreate                  hipMemPoolCreate
#define gpuMemPoolDestroy                 hipMemPoolDestroy
#define gpuMemAllocFromPoolAsync          hipMallocFromPoolAsync
#define gpuMemFreeAsync                   hipFreeAsync
#define gpuFreeAsync                      hipFreeAsync
#define gpuGetStreamPriorityRange         hipDeviceGetStreamPriorityRange
#define gpuStreamCreateWithPriority       hipStreamCreateWithPriority
#define gpuStreamQuery                    hipStreamQuery
#define gpuStreamDestroy                  hipStreamDestroy
#define gpuEventSynchronize               hipEventSynchronize
#define gpuEventRecord                    hipEventRecord
#define gpuStreamWaitEvent                hipStreamWaitEvent
#define gpuEventQuery                     hipEventQuery
#define gpuEventDestroy                   hipEventDestroy
#define gpuEventCreate                    hipEventCreateWithFlags
#define gpuMemcpy2DAsync                  hipMemcpyParam2DAsync
#define gpuMemcpy2D                       hipMemcpyParam2D
#define gpuMemcpyDtoH                     hipMemcpyDtoH
#define gpuMemcpyDtoDAsync                hipMemcpyDtoDAsync
#define gpuMemcpyDtoD                     hipMemcpyDtoD
#define gpuMemcpy                         hipMemcpy
#define gpuGetErrorName                   hipDrvGetErrorName
#define gpuGetErrorString                 hipDrvGetErrorString
#define GPUrtGetErrorName                 hipGetErrorName
#define GPUrtGetErrorString               hipGetErrorString
#define gpuGetLastError                   hipGetLastError
#define gpuInit                           hipInit
#define gpuDeviceGetCount                 hipGetDeviceCount
#define gpuDeviceGet                      hipDeviceGet
#define gpuCtxCreate                      hipCtxCreate
#define gpuCtxDestroy                     hipCtxDestroy
#define gpuDevicePrimaryCtxRetain         hipDevicePrimaryCtxRetain
#define gpuDevicePrimaryCtxRelease        hipDevicePrimaryCtxRelease
#define gpuCtxPushCurrent                 hipCtxPushCurrent
#define gpuCtxPopCurrent                  hipCtxPopCurrent
#define GPUrtLaunchKernel                 hipLaunchKernel
#define gpuMemPoolTrimTo                  hipMemPoolTrimTo
#define gpuDeviceGetDefaultMemPool        hipDeviceGetDefaultMemPool
#define gpuMemPoolSetAttribute            hipMemPoolSetAttribute
#define gpuPointerGetAttribute            hipPointerGetAttribute
#define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
#define gpuDeviceSynchronize              hipDeviceSynchronize
#define gpuEventElapsedTime               hipEventElapsedTime
#define gpuMemcpyHostToHost               hipMemcpyHostToHost
#define gpuMemcpyHostToDevice             hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost             hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice           hipMemcpyDeviceToDevice
#define gpuMemcpyDefault                  hipMemcpyDefault
#define gpuMemPoolAttrReleaseThreshold    hipMemPoolAttrReleaseThreshold
GPUresult gpuMemcpyAsync(GPUdeviceptr, GPUdeviceptr, size_t, GPUstream_t);
GPUresult gpuMemcpyHtoDAsync(GPUdeviceptr, const void*, size_t, GPUstream_t);
GPUresult gpuMemcpyHtoD(GPUdeviceptr, const void*, size_t);
GPUresult gpuMemcpyPeerAsync(
    GPUdeviceptr,
    GPUcontext,
    GPUdevice,
    GPUdeviceptr,
    GPUcontext,
    GPUdevice,
    size_t,
    GPUstream_t
);

// rocBLAS types
using blasStatus_t = rocblas_status;
using blasHandle_t = rocblas_handle;

// rocBLAS functions
#define blasCreate          rocblas_create_handle
#define blasSetStream       rocblas_set_stream
#define blasDestroy         rocblas_destroy_handle
#define blasGetStatusString rocblas_status_to_string
const char* blasGetStatusName(blasStatus_t);

}  // namespace kmm
