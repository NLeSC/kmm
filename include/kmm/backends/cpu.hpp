#pragma once

#include <cstddef>

#include "kmm/utils/macros.hpp"

namespace kmm {

// Types
using half_type = unsigned char;
using bfloat16_type = char;
using size_t = std::size_t;
using GPUdevice = int;
class dim3 {
  public:
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}

    unsigned int x;
    unsigned int y;
    unsigned int z;
};
enum GPUmemAllocationType { GPU_MEM_ALLOCATION_TYPE_PINNED = 1 };
enum GPUmemAllocationHandleType { GPU_MEM_HANDLE_TYPE_NONE = 0 };
enum GPUmemLocationType { GPU_MEM_LOCATION_TYPE_DEVICE = 1 };
struct GPUmemLocation {
    int id;
    GPUmemLocationType type;
};
struct GPUmemPoolProps {
    GPUmemAllocationType allocType;
    GPUmemAllocationHandleType handleTypes;
    GPUmemLocation location;
    size_t maxSize;
    unsigned char reserved[54];
    unsigned short usage;
    void* win32SecurityAttributes;
};
using GPUcontext = int*;
using GPUstream_t = int*;
using GPUdeviceptr = unsigned long long;
using GPUmemoryPool = int*;
using GPUevent_t = void*;
enum GPUmemorytype { GPU_MEMORYTYPE_HOST, GPU_MEMORYTYPE_DEVICE };
struct GPU_MEMCPY2D {
    size_t Height;
    size_t WidthInBytes;
    GPUdeviceptr dstDevice;
    void* dstHost;
    GPUmemorytype dstMemoryType;
    size_t dstPitch;
    size_t dstXInBytes;
    size_t dstY;
    GPUdeviceptr srcDevice;
    const void* srcHost;
    GPUmemorytype srcMemoryType;
    size_t srcPitch;
    size_t srcXInBytes;
    size_t srcY;
};
enum GPUresult {};
enum gpuError_t {};
enum gpuMemcpyKind {};
enum gpuMemPoolAttr {};
#define gpuMemPoolAttrReleaseThreshold gpuMemPoolAttr(3)
using GPUdevice_attribute = int;
using GPUstream_flags = int;
using GPUevent_wait_flags = int;
enum GPUpointer_attribute {};

// Constants
#define GPU_DEVICE_ATTRIBUTE_MAX                      1
#define GPU_MEMHOSTALLOC_PORTABLE                     0
#define GPU_MEMHOSTALLOC_DEVICEMAP                    0
#define GPU_SUCCESS                                   0
#define GPU_ERROR_OUT_OF_MEMORY                       0
#define GPU_ERROR_UNKNOWN                             0
#define GPU_ERROR_NOT_READY                           0
#define GPU_EVENT_DISABLE_TIMING                      2
#define GPU_ERROR_NO_DEVICE                           100
#define GPU_CTX_MAP_HOST                              0x08
#define gpuMemcpyHostToHost                           0
#define gpuMemcpyHostToDevice                         1
#define gpuMemcpyDeviceToHost                         2
#define gpuMemcpyDeviceToDevice                       3
#define gpuMemcpyDefault                              4
#define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    GPUdevice_attribute(1)
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          GPUdevice_attribute(2)
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          GPUdevice_attribute(3)
#define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          GPUdevice_attribute(4)
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           GPUdevice_attribute(5)
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           GPUdevice_attribute(6)
#define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           GPUdevice_attribute(7)
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR GPUdevice_attribute(75)
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR GPUdevice_attribute(76)
#define GPU_STREAM_NON_BLOCKING                       GPUstream_flags(1)
#define GPU_EVENT_WAIT_DEFAULT                        GPUevent_wait_flags(0)
#define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE             GPUpointer_attribute(2)
#define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL          GPUpointer_attribute(9)

// Dummy functions
GPUresult gpuCtxGetDevice(GPUdevice*);
GPUresult gpuDeviceGetName(char*, int, GPUdevice);
GPUresult gpuDeviceGetAttribute(int*, GPUdevice_attribute, GPUdevice);
GPUresult gpuMemGetInfo(size_t*, size_t*);
GPUresult gpuMemcpyHtoDAsync(GPUdeviceptr, const void*, size_t, GPUstream_t);
GPUresult gpuMemcpyDtoHAsync(void*, GPUdeviceptr, size_t, GPUstream_t);
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
GPUresult gpuStreamSynchronize(GPUstream_t);
GPUresult gpuMemsetD8Async(GPUdeviceptr, unsigned char, size_t, GPUstream_t);
GPUresult gpuMemsetD16Async(GPUdeviceptr, unsigned short, size_t, GPUstream_t);
GPUresult gpuMemsetD32Async(GPUdeviceptr, unsigned int, size_t, GPUstream_t);
gpuError_t gpuMemsetAsync(void* dst, int value, size_t sizeBytes, GPUstream_t stream);
gpuError_t gpuMemcpy(void*, const void*, size_t, enum gpuMemcpyKind);
GPUresult gpuMemcpyAsync(GPUdeviceptr, GPUdeviceptr, size_t, GPUstream_t);
GPUresult gpuMemHostAlloc(void**, size_t, unsigned int);
GPUresult gpuMemFreeHost(void*);
GPUresult gpuMemAlloc(GPUdeviceptr*, size_t);
GPUresult gpuMemAllocAsync(GPUdeviceptr*, size_t, GPUstream_t);
gpuError_t gpuMalloc(GPUdeviceptr*, size_t);
GPUresult gpuMemFree(GPUdeviceptr);
gpuError_t gpuFree(GPUdeviceptr);
GPUresult gpuMemPoolCreate(GPUmemoryPool*, const GPUmemPoolProps*);
GPUresult gpuMemPoolDestroy(GPUmemoryPool);
GPUresult gpuMemAllocFromPoolAsync(GPUdeviceptr*, size_t, GPUmemoryPool, GPUstream_t);
GPUresult gpuMemFreeAsync(GPUdeviceptr, GPUstream_t);
gpuError_t gpuFreeAsync(GPUdeviceptr, GPUstream_t);
GPUresult gpuGetStreamPriorityRange(int*, int*);
GPUresult gpuStreamCreateWithPriority(GPUstream_t*, unsigned int, int);
GPUresult gpuStreamQuery(GPUstream_t);
GPUresult gpuStreamDestroy(GPUstream_t);
GPUresult gpuEventSynchronize(GPUevent_t);
GPUresult gpuEventRecord(GPUevent_t, GPUstream_t);
GPUresult gpuStreamWaitEvent(GPUstream_t, GPUevent_t, unsigned int);
GPUresult gpuEventQuery(GPUevent_t);
GPUresult gpuEventDestroy(GPUevent_t);
GPUresult gpuEventCreate(GPUevent_t, unsigned int);
GPUresult gpuMemcpyHtoD(GPUdeviceptr, const void*, size_t);
GPUresult gpuMemcpy2DAsync(const GPU_MEMCPY2D*, GPUstream_t);
GPUresult gpuMemcpy2D(const GPU_MEMCPY2D*);
GPUresult gpuMemcpyDtoH(void*, GPUdeviceptr, size_t);
GPUresult gpuMemcpyDtoDAsync(GPUdeviceptr, GPUdeviceptr, size_t, GPUstream_t);
GPUresult gpuMemcpyDtoD(GPUdeviceptr, GPUdeviceptr, size_t);
GPUresult gpuGetErrorName(GPUresult, const char**);
GPUresult gpuGetErrorString(GPUresult, const char**);
const char* GPUrtGetErrorName(gpuError_t);
const char* GPUrtGetErrorString(gpuError_t);
gpuError_t gpuGetLastError(void);
GPUresult gpuInit(unsigned int);
GPUresult gpuDeviceGetCount(int*);
GPUresult gpuDeviceGet(GPUdevice*, int);
GPUresult gpuPointerGetAttribute(void*, GPUpointer_attribute, GPUdeviceptr);
GPUresult gpuCtxCreate(GPUcontext*, unsigned int, GPUdevice);
GPUresult gpuCtxDestroy(GPUcontext);
GPUresult gpuDevicePrimaryCtxRetain(GPUcontext*, GPUdevice);
GPUresult gpuDevicePrimaryCtxRelease(GPUdevice);
GPUresult gpuCtxPushCurrent(GPUcontext);
GPUresult gpuCtxPopCurrent(GPUcontext*);
gpuError_t GPUrtLaunchKernel(const void*, dim3, dim3, void**, size_t, GPUstream_t);
GPUresult gpuMemPoolTrimTo(GPUmemoryPool, size_t);
GPUresult gpuDeviceGetDefaultMemPool(GPUmemoryPool*, GPUdevice);
GPUresult gpuMemPoolSetAttribute(GPUmemoryPool, gpuMemPoolAttr, void*);
gpuError_t gpuDeviceSynchronize();
gpuError_t gpuEventElapsedTime(float* ms, GPUevent_t start, GPUevent_t stop);

// Dummy atomic operations
template<typename T>
T atomicAnd(T* input, T output) {
    return 0;
};
template<typename T>
T atomicOr(T* input, T output) {
    return 0;
};
template<typename T>
T atomicAdd(T* input, T output) {
    return 0;
};
template<typename T>
T atomicMin(T* input, T output) {
    return 0;
};
template<typename T>
T atomicMax(T* input, T output) {
    return 0;
};

// Dummy BLAS types
using blasHandle_t = void*;
enum blasStatus_t {};

// Dummy BLAS functions
blasStatus_t blasCreate(blasHandle_t);
blasStatus_t blasSetStream(blasHandle_t, GPUstream_t);
blasStatus_t blasDestroy(blasHandle_t);
const char* blasGetStatusName(blasStatus_t);
const char* blasGetStatusString(blasStatus_t);

}  // namespace kmm
