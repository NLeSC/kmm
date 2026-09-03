#pragma once

#include <cstddef>

#include "kmm/core/macros.hpp"

namespace kmm {

// Types

using half_type = unsigned char;
using bfloat16_type = char;
using size_t = std::size_t;
using g_device_t = int;

class dim3 {
  public:
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

enum g_mem_allocation_type_t { G_MEM_ALLOCATION_TYPE_PINNED = 1 };
enum g_mem_allocation_handle_type_t { G_MEM_HANDLE_TYPE_NONE = 0 };
enum g_mem_location_type_t { G_MEM_LOCATION_TYPE_DEVICE = 1 };

struct g_mem_location_t {
    int id;
    g_mem_location_type_t type;
};

struct g_mem_pool_props_t {
    g_mem_allocation_type_t allocType;
    g_mem_allocation_handle_type_t handleTypes;
    g_mem_location_t location;
    size_t maxSize;
    unsigned char reserved[54];
    unsigned short usage;
    void* win32SecurityAttributes;
};

using g_context_t = int*;
using gpu_stream_t = int*;
using g_stream_t = int*;
using g_device_ptr_t = unsigned long long;
using gpu_memory_pool_t = int*;
using g_memory_pool_t = int*;
using gpu_event_t = void*;
using g_event_t = void*;

enum g_memory_type_t { G_MEMORYTYPE_HOST, G_MEMORYTYPE_DEVICE };

struct gpu_memcpy2d_t {
    size_t Height;
    size_t WidthInBytes;
    g_device_ptr_t dstDevice;
    void* dstHost;
    g_memory_type_t dstMemoryType;
    size_t dstPitch;
    size_t dstXInBytes;
    size_t dstY;
    g_device_ptr_t srcDevice;
    const void* srcHost;
    g_memory_type_t srcMemoryType;
    size_t srcPitch;
    size_t srcXInBytes;
    size_t srcY;
};

struct gpu_mem_pool_t {};

enum g_result_t {};
enum gpu_error_t {};
enum gpu_memcpy_kind_t {};
enum gpu_mem_pool_attr_t {};

using g_device_attribute_t = int;
using g_stream_flags_t = int;
using g_event_wait_flags_t = int;
enum g_pointer_attribute_t {};

// Device Management Constants & Functions

#define G_DEVICE_ATTRIBUTE_MAX                      1
#define G_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    g_device_attribute_t(1)
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          g_device_attribute_t(2)
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          g_device_attribute_t(3)
#define G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          g_device_attribute_t(4)
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           g_device_attribute_t(5)
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           g_device_attribute_t(6)
#define G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           g_device_attribute_t(7)
#define G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR g_device_attribute_t(75)
#define G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR g_device_attribute_t(76)
#define GPU_ERROR_PEER_ACCESS_ALREADY_ENABLED       704

g_result_t g_init(unsigned int);
g_result_t g_device_get_count(int*);
g_result_t g_device_get(g_device_t*, int);
g_result_t g_device_get_name(char*, int, g_device_t);
g_result_t g_device_total_mem(size_t*, g_device_t);
g_result_t g_device_get_attribute(int*, g_device_attribute_t, g_device_t);
g_result_t g_device_can_access_peer(int*, g_device_t, g_device_t);
g_result_t g_ctx_get_device(g_device_t*);
g_result_t g_ctx_get_id(g_context_t, unsigned long long*);
g_result_t g_ctx_create(g_context_t*, unsigned int, g_device_t);
g_result_t g_ctx_destroy(g_context_t);
g_result_t g_device_primary_ctx_retain(g_context_t*, g_device_t);
g_result_t g_device_primary_ctx_release(g_device_t);
g_result_t g_ctx_push_current(g_context_t);
g_result_t g_ctx_pop_current(g_context_t*);
g_result_t g_ctx_enable_peer_access(g_context_t, unsigned int);

// Stream Management Constants & Functions

#define G_STREAM_NON_BLOCKING g_stream_flags_t(1)
#define G_EVENT_WAIT_DEFAULT  g_event_wait_flags_t(0)

g_result_t g_ctx_get_stream_priority_range(int*, int*);
g_result_t g_stream_create(g_stream_t*, unsigned int);
g_result_t g_stream_create_with_priority(g_stream_t*, unsigned int, int);
g_result_t g_stream_get_device(g_stream_t, g_device_t*);
g_result_t g_stream_get_id(g_stream_t, unsigned long long*);
g_result_t g_stream_get_ctx(g_stream_t, g_context_t*);
g_result_t g_stream_query(g_stream_t);
g_result_t g_stream_synchronize(g_stream_t);
g_result_t g_stream_destroy(g_stream_t);
g_result_t g_stream_wait_event(g_stream_t, g_event_t, unsigned int);

// Event Management Constants & Functions

#define G_EVENT_DISABLE_TIMING 2

gpu_error_t gpu_event_create(gpu_event_t);
g_result_t g_event_create(g_event_t, unsigned int);
gpu_error_t gpu_event_record(gpu_event_t, gpu_stream_t = 0);
g_result_t g_event_record(g_event_t, g_stream_t);
g_result_t g_event_query(g_event_t);
gpu_error_t gpu_event_synchronize(gpu_event_t);
g_result_t g_event_synchronize(g_event_t);
gpu_error_t gpu_event_destroy(gpu_event_t);
g_result_t g_event_destroy(g_event_t);
gpu_error_t gpu_event_elapsed_time(float* ms, g_event_t start, g_event_t stop);

// Memory Management Constants & Functions

#define G_MEMHOSTALLOC_PORTABLE            0
#define G_MEMHOSTALLOC_DEVICEMAP           0
#define G_MEMORYTYPE_HOST                  G_MEMORYTYPE_HOST
#define G_MEMORYTYPE_DEVICE                G_MEMORYTYPE_DEVICE
#define G_POINTER_ATTRIBUTE_MEMORY_TYPE    g_pointer_attribute_t(2)
#define G_POINTER_ATTRIBUTE_DEVICE_ORDINAL g_pointer_attribute_t(9)
#define G_MEM_ATTACH_GLOBAL                1

g_result_t g_mem_get_info(size_t*, size_t*);
g_result_t gpu_mem_get_info(size_t*, size_t*);
g_result_t g_mem_alloc(g_device_ptr_t*, size_t);
g_result_t g_mem_free(g_device_ptr_t);
gpu_error_t gpu_malloc(g_device_ptr_t*, size_t);
gpu_error_t gpu_free(g_device_ptr_t);
g_result_t g_mem_host_alloc(void**, size_t, unsigned int);
g_result_t g_mem_free_host(void*);
g_result_t g_pointer_get_attribute(void*, g_pointer_attribute_t, g_device_ptr_t);
g_result_t g_mem_alloc_managed(g_device_ptr_t*, size_t, unsigned int);
g_result_t g_mem_host_get_device_pointer(g_device_ptr_t*, void*, unsigned int);
g_result_t g_mem_prefetch_async(g_device_ptr_t, size_t, int, g_stream_t);

// Memory Copy Operations

#define gpu_memcpy_host_to_host     0
#define gpu_memcpy_host_to_device   1
#define gpu_memcpy_device_to_host   2
#define gpu_memcpy_device_to_device 3
#define gpu_memcpy_default          4

gpu_error_t gpu_memcpy(void*, const void*, size_t, enum gpu_memcpy_kind_t);
g_result_t g_memcpy_h_to_d(g_device_ptr_t, const void*, size_t);
g_result_t g_memcpy_h_to_d_async(g_device_ptr_t, const void*, size_t, g_stream_t);
g_result_t g_memcpy_d_to_h(void*, g_device_ptr_t, size_t);
g_result_t g_memcpy_d_to_h_async(void*, g_device_ptr_t, size_t, g_stream_t);
g_result_t g_memcpy_d_to_d(g_device_ptr_t, g_device_ptr_t, size_t);
g_result_t g_memcpy_d_to_d_async(g_device_ptr_t, g_device_ptr_t, size_t, g_stream_t);
g_result_t g_memcpy_async(g_device_ptr_t, g_device_ptr_t, size_t, g_stream_t);
g_result_t g_memcpy_peer_async(
    g_device_ptr_t,
    g_context_t,
    g_device_t,
    g_device_ptr_t,
    g_context_t,
    g_device_t,
    size_t,
    g_stream_t
);

// Memory Fill Operations

g_result_t g_memset_d8_async(g_device_ptr_t, unsigned char, size_t, g_stream_t);
g_result_t g_memset_d16_async(g_device_ptr_t, unsigned short, size_t, g_stream_t);
g_result_t g_memset_d32_async(g_device_ptr_t, unsigned int, size_t, g_stream_t);
gpu_error_t gpu_memset_async(void* dst, int value, size_t sizeBytes, g_stream_t stream);

// 2D Memory Operations

g_result_t g_memcpy_2d(const gpu_memcpy2d_t*);
g_result_t g_memcpy_2d_async(const gpu_memcpy2d_t*, g_stream_t);

g_result_t g_memset_2d_d8_async(g_device_ptr_t, size_t, unsigned char, size_t, size_t, g_stream_t);
g_result_t g_memset_2d_d16_async(
    g_device_ptr_t,
    size_t,
    unsigned short,
    size_t,
    size_t,
    g_stream_t
);
g_result_t g_memset_2d_d32_async(g_device_ptr_t, size_t, unsigned int, size_t, size_t, g_stream_t);

// Memory Pool Management Constants & Functions

#define G_MEM_ALLOCATION_TYPE_PINNED      G_MEM_ALLOCATION_TYPE_PINNED
#define G_MEM_HANDLE_TYPE_NONE            G_MEM_HANDLE_TYPE_NONE
#define G_MEM_LOCATION_TYPE_DEVICE        G_MEM_LOCATION_TYPE_DEVICE
#define G_CTX_MAP_HOST                    0x08
#define g_mem_pool_attr_release_threshold gpu_mem_pool_attr_t(3)

gpu_error_t gpu_malloc_async(void**, size_t, g_stream_t);
g_result_t g_mem_pool_create(g_memory_pool_t*, const g_mem_pool_props_t*);
g_result_t g_mem_pool_destroy(g_memory_pool_t);
g_result_t g_mem_alloc_from_pool_async(g_device_ptr_t*, size_t, g_memory_pool_t, g_stream_t);
g_result_t g_mem_alloc_async(g_device_ptr_t*, size_t, g_stream_t);
g_result_t g_mem_free_async(g_device_ptr_t, g_stream_t);
gpu_error_t gpu_free_async(g_device_ptr_t, g_stream_t);
g_result_t g_mem_pool_trim_to(g_memory_pool_t, size_t);
gpu_error_t gpu_device_get_default_mem_pool(g_memory_pool_t*, int);
g_result_t g_device_get_default_mem_pool(g_memory_pool_t*, int);
gpu_error_t gpu_mem_pool_set_attribute(gpu_memory_pool_t, gpu_mem_pool_attr_t, void*);
g_result_t g_mem_pool_set_attribute(g_memory_pool_t, gpu_mem_pool_attr_t, void*);

// Error Handling Constants & Functions

#define G_SUCCESS             0
#define G_ERROR_OUT_OF_MEMORY 0
#define G_ERROR_UNKNOWN       0
#define G_ERROR_NOT_READY     0
#define G_ERROR_NO_DEVICE     100

#define GPU_SUCCESS             0
#define GPU_ERROR_OUT_OF_MEMORY 0
#define GPU_ERROR_UNKNOWN       0
#define GPU_ERROR_NOT_READY     0
#define GPU_ERROR_NO_DEVICE     100

g_result_t g_get_error_name(g_result_t, const char**);
g_result_t g_get_error_string(g_result_t, const char**);
const char* gpu_get_error_name(gpu_error_t);
const char* gpu_get_error_string(gpu_error_t);
gpu_error_t gpu_get_last_error(void);

// Kernel Execution

gpu_error_t gpu_launch_kernel(const void*, dim3, dim3, void**, size_t, g_stream_t);
gpu_error_t gpu_device_synchronize();

// Atomic Operations

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

// Dummy BLAS Support

using blas_handle_t = void*;
enum blas_status_t {};

blas_status_t blas_create(blas_handle_t);
blas_status_t blas_set_stream(blas_handle_t, g_stream_t);
blas_status_t blas_destroy(blas_handle_t);
const char* blas_get_status_name(blas_status_t);
const char* blas_get_status_string(blas_status_t);

}  // namespace kmm
