#pragma once

#include <cstddef>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "kmm/utils/macros.hpp"

namespace kmm {

using half_type = __half;
using bfloat16_type = __nv_bfloat16;

#define GPU_DEVICE_ATTRIBUTE_MAX                      CU_DEVICE_ATTRIBUTE_MAX
#define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define GPU_MEMHOSTALLOC_PORTABLE                     CU_MEMHOSTALLOC_PORTABLE
#define GPU_MEMHOSTALLOC_DEVICEMAP                    CU_MEMHOSTALLOC_DEVICEMAP
#define GPU_SUCCESS                                   CUDA_SUCCESS
#define GPU_ERROR_OUT_OF_MEMORY                       CUDA_ERROR_OUT_OF_MEMORY
#define GPU_MEM_ALLOCATION_TYPE_PINNED                CU_MEM_ALLOCATION_TYPE_PINNED
#define GPU_MEM_HANDLE_TYPE_NONE                      CU_MEM_HANDLE_TYPE_NONE
#define GPU_MEM_LOCATION_TYPE_DEVICE                  CU_MEM_LOCATION_TYPE_DEVICE
#define GPU_ERROR_UNKNOWN                             CUDA_ERROR_UNKNOWN
#define GPU_STREAM_NON_BLOCKING                       CU_STREAM_NON_BLOCKING
#define GPU_EVENT_WAIT_DEFAULT                        CU_EVENT_WAIT_DEFAULT
#define GPU_ERROR_NOT_READY                           CUDA_ERROR_NOT_READY
#define GPU_EVENT_DISABLE_TIMING                      CU_EVENT_DISABLE_TIMING
#define GPU_MEMORYTYPE_HOST                           CU_MEMORYTYPE_HOST
#define GPU_MEMORYTYPE_DEVICE                         CU_MEMORYTYPE_DEVICE
#define GPU_ERROR_NO_DEVICE                           CUDA_ERROR_NO_DEVICE
#define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE             CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL          CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define GPU_CTX_MAP_HOST                              CU_CTX_MAP_HOST
#define gpuCtxGetDevice                               cuCtxGetDevice
#define gpuDeviceGetName                              cuDeviceGetName
#define gpuDeviceGetAttribute                         cuDeviceGetAttribute
#define gpuMemGetInfo                                 cuMemGetInfo
#define gpuMemcpyHtoDAsync                            cuMemcpyHtoDAsync
#define gpuMemcpyDtoHAsync                            cuMemcpyDtoHAsync
#define gpuStreamSynchronize                          cuStreamSynchronize
#define gpuMemsetD8Async                              cuMemsetD8Async
#define gpuMemsetD16Async                             cuMemsetD16Async
#define gpuMemsetD32Async                             cuMemsetD32Async
#define gpuMemsetAsync                                cudaMemsetAsync
#define gpuMemcpy                                     cudaMemcpy
#define gpuMemcpyAsync                                cuMemcpyAsync
#define gpuMemHostAlloc                               cuMemHostAlloc
#define gpuMemFreeHost                                cuMemFreeHost
#define gpuMemAlloc                                   cuMemAlloc
#define gpuMemAllocAsync                              cuMemAllocAsync
#define gpuAlloc                                      cudaMalloc
#define gpuMemFree                                    cuMemFree
#define gpuFree                                       cudaFree
#define gpuMemPoolCreate                              cuMemPoolCreate
#define gpuMemPoolDestroy                             cuMemPoolDestroy
#define gpuMemAllocFromPoolAsync                      cuMemAllocFromPoolAsync
#define gpuMemFreeAsync                               cuMemFreeAsync
#define gpuFreeAsync                                  cudaFreeAsync
#define gpuGetStreamPriorityRange                     cuCtxGetStreamPriorityRange
#define gpuStreamCreateWithPriority                   cuStreamCreateWithPriority
#define gpuStreamQuery                                cuStreamQuery
#define gpuStreamDestroy                              cuStreamDestroy
#define gpuEventSynchronize                           cuEventSynchronize
#define gpuEventRecord                                cuEventRecord
#define gpuStreamWaitEvent                            cuStreamWaitEvent
#define gpuEventQuery                                 cuEventQuery
#define gpuEventDestroy                               cuEventDestroy
#define gpuEventCreate                                cuEventCreate
#define gpuMemcpyHtoD                                 cuMemcpyHtoD
#define gpuMemcpy2DAsync                              cuMemcpy2DAsync
#define gpuMemcpy2D                                   cuMemcpy2D
#define gpuMemcpyDtoH                                 cuMemcpyDtoH
#define gpuMemcpyDtoDAsync                            cuMemcpyDtoDAsync
#define gpuMemcpyDtoD                                 cuMemcpyDtoD
#define gpuGetErrorName                               cuGetErrorName
#define gpuGetErrorString                             cuGetErrorString
#define GPUrtGetErrorName                             cudaGetErrorName
#define GPUrtGetErrorString                           cudaGetErrorString
#define gpuGetLastError                               cudaGetLastError
#define gpuInit                                       cuInit
#define gpuDeviceGetCount                             cuDeviceGetCount
#define gpuDeviceGet                                  cuDeviceGet
#define gpuCtxCreate                                  cuCtxCreate
#define gpuCtxDestroy                                 cuCtxDestroy
#define gpuDevicePrimaryCtxRetain                     cuDevicePrimaryCtxRetain
#define gpuDevicePrimaryCtxRelease                    cuDevicePrimaryCtxRelease
#define gpuCtxPushCurrent                             cuCtxPushCurrent
#define gpuCtxPopCurrent                              cuCtxPopCurrent
#define GPUrtLaunchKernel                             cudaLaunchKernel
#define gpuMemPoolTrimTo                              cuMemPoolTrimTo
#define gpuDeviceGetDefaultMemPool                    cuDeviceGetDefaultMemPool
#define gpuMemPoolSetAttribute                        cuMemPoolSetAttribute
#define gpuPointerGetAttribute                        cuPointerGetAttribute
#define gpuDeviceSynchronize                          cudaDeviceSynchronize
#define gpuEventElapsedTime                           cudaEventElapsedTime
#define gpuMemcpyHostToHost                           cudaMemcpyHostToHost
#define gpuMemcpyHostToDevice                         cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost                         cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice                       cudaMemcpyDeviceToDevice
#define gpuMemcpyDefault                              cudaMemcpyDefault
#define gpuMemPoolAttrReleaseThreshold                cuMemPoolAttrReleaseThreshold

using GPUresult = CUresult;
using gpuError_t = cudaError_t;
using GPUdevice = CUdevice;
using GPUdevice_attribute = CUdevice_attribute;
using GPUcontext = CUcontext;
using GPUmemorytype = CUmemorytype;
using GPUstream_t = CUstream;
using GPUdeviceptr = CUdeviceptr;
using GPUmemoryPool = CUmemoryPool;
using GPUmemPoolProps = CUmemPoolProps;
using GPUmemAllocationType = CUmemAllocationType;
using GPUmemAllocationHandleType = CUmemAllocationHandleType;
using GPUmemLocationType = CUmemLocationType;
using GPUevent_t = CUevent;
using GPU_MEMCPY2D = CUDA_MEMCPY2D;
using gpuMemPoolAttr = cudaMemPoolAttr;

// cuBLAS
#define blasCreate          cublasCreate
#define blasSetStream       cublasSetStream
#define blasDestroy         cublasDestroy
#define blasGetStatusName   cublasGetStatusName
#define blasGetStatusString cublasGetStatusString

using blasStatus_t = cublasStatus_t;
using blasHandle_t = cublasHandle_t;

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

}  // namespace kmm
