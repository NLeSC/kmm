#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/runtime/memops/types.hpp"
#include "kmm/utils/backends.hpp"

namespace kmm {

g_result_t g_ctx_get_device(g_device_t* device) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_ctx_get_id(g_context_t ctx, unsigned long long* ctxId) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_device_get_name(char* name, int len, g_device_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_device_total_mem(size_t*, g_device_t) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_device_get_attribute(int* value, g_device_attribute_t attribute, g_device_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_device_can_access_peer(int* canAccessPeer, g_device_t dev, g_device_t peerDev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_get_info(size_t* free, size_t* total) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t gpu_mem_get_info(size_t* free, size_t* total) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_h_to_d_async(
    g_device_ptr_t dev,
    const void* host,
    size_t size,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_d_to_h_async(
    void* host,
    g_device_ptr_t device,
    size_t size,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_peer_async(
    g_device_ptr_t dst_ptr,
    g_context_t dst_ctx,
    g_device_t dst_id,
    g_device_ptr_t std_ptr,
    g_context_t src_ctx,
    g_device_t src_id,
    size_t size,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_synchronize(g_stream_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memset_d8_async(
    g_device_ptr_t dev,
    unsigned char val,
    size_t size,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memset_d16_async(
    g_device_ptr_t dev,
    unsigned short val,
    size_t size,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memset_d32_async(
    g_device_ptr_t dev,
    unsigned int val,
    size_t size,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_memset_async(void* dst, int value, size_t sizeBytes, g_stream_t stream) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

gpu_error_t gpu_memcpy(void* dst, const void* src, size_t count, enum gpu_memcpy_kind_t kind) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_memcpy_async(g_device_ptr_t dst, g_device_ptr_t src, size_t size, g_stream_t stream) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_host_alloc(void** host, size_t size, unsigned int flags) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_free_host(void* dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_alloc(g_device_ptr_t* dev, size_t size) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_alloc_async(g_device_ptr_t*, size_t, g_stream_t) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_malloc(g_device_ptr_t*, size_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_mem_free(g_device_ptr_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_free(g_device_ptr_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

gpu_error_t gpu_malloc_async(void**, size_t, g_stream_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_mem_pool_create(g_memory_pool_t* pool, const g_mem_pool_props_t* props) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_pool_destroy(g_memory_pool_t pool) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_alloc_from_pool_async(
    g_device_ptr_t* dev,
    size_t size,
    g_memory_pool_t pool,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_free_async(g_device_ptr_t dev, g_stream_t stream) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_free_async(g_device_ptr_t, g_stream_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_ctx_get_stream_priority_range(int* least, int* greatest) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_create(g_stream_t* stream, unsigned int flags) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_create_with_priority(g_stream_t* stream, unsigned int flags, int priority) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_get_device(g_stream_t hStream, g_device_t* device) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_get_id(g_stream_t hStream, unsigned long long* streamId) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_get_ctx(g_stream_t hStream, g_context_t* pctx) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_query(g_stream_t stream) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_destroy(g_stream_t stream) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_event_synchronize(gpu_event_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_event_synchronize(g_event_t event) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_event_record(gpu_event_t, gpu_stream_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_event_record(g_event_t event, g_stream_t stream) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_stream_wait_event(g_stream_t stream, g_event_t event, unsigned int flags) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_event_query(g_event_t event) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_event_destroy(gpu_event_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_event_destroy(g_event_t event) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_event_create(gpu_event_t) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_event_create(g_event_t event, unsigned int flags) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_h_to_d(g_device_ptr_t dest, const void* src, size_t size) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_2d_async(const gpu_memcpy2d_t* dev, g_stream_t stream) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_2d(const gpu_memcpy2d_t* dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_d_to_h(void* dest, g_device_ptr_t src, size_t size) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_d_to_d_async(
    g_device_ptr_t dest,
    g_device_ptr_t src,
    size_t size,
    g_stream_t stream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_memcpy_d_to_d(g_device_ptr_t dest, g_device_ptr_t src, size_t size) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_get_error_name(g_result_t error, const char** name) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_get_error_string(g_result_t error, const char** desc) {
    return g_result_t(G_ERROR_UNKNOWN);
}

const char* gpu_get_error_name(gpu_error_t error) {
    return "";
}

const char* gpu_get_error_string(gpu_error_t error) {
    return "";
}

gpu_error_t gpu_get_last_error(void) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_init(unsigned int flags) {
    return g_result_t(G_ERROR_NO_DEVICE);
}

g_result_t g_device_get_count(int* count) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_device_get(g_device_t* dev, int ordinal) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_pointer_get_attribute(void* prt, g_pointer_attribute_t attribute, g_device_ptr_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_alloc_managed(g_device_ptr_t* dptr, size_t bytesize, unsigned int flags) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_host_get_device_pointer(g_device_ptr_t* pdptr, void* p, unsigned int Flags) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_mem_prefetch_async(
    g_device_ptr_t devPtr,
    size_t count,
    int device,
    g_stream_t hStream
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_ctx_create(g_context_t* ctx, unsigned int flags, g_device_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_ctx_destroy(g_context_t ctx) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_device_primary_ctx_retain(g_context_t* ctx, g_device_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_device_primary_ctx_release(g_device_t dev) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_ctx_push_current(g_context_t ctx) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_ctx_pop_current(g_context_t* ctx) {
    return g_result_t(G_ERROR_UNKNOWN);
}

g_result_t g_ctx_enable_peer_access(g_context_t peerContext, unsigned int Flags) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_launch_kernel(
    const void* func,
    dim3 grid,
    dim3 block,
    void** args,
    size_t smem,
    g_stream_t stream
) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_mem_pool_trim_to(g_memory_pool_t pool, size_t size) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_device_get_default_mem_pool(g_memory_pool_t*, int) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_device_get_default_mem_pool(g_memory_pool_t*, int) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_mem_pool_set_attribute(gpu_memory_pool_t, gpu_mem_pool_attr_t, void*) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

g_result_t g_mem_pool_set_attribute(
    g_memory_pool_t memPool,
    gpu_mem_pool_attr_t attr,
    void* value
) {
    return g_result_t(G_ERROR_UNKNOWN);
}

gpu_error_t gpu_device_synchronize() {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

gpu_error_t gpu_event_elapsed_time(float* ms, g_event_t start, g_event_t stop) {
    return gpu_error_t(GPU_ERROR_UNKNOWN);
}

blas_status_t blas_create(blas_handle_t blas) {
    return blas_status_t(1);
}

blas_status_t blas_set_stream(blas_handle_t blas, g_stream_t stream) {
    return blas_status_t(1);
}

blas_status_t blas_destroy(blas_handle_t blas) {
    return blas_status_t(1);
}

const char* blas_get_status_name(blas_status_t blas) {
    return "";
}

const char* blas_get_status_string(blas_status_t blas) {
    return "";
}

void execute_gpu_fill_async(
    g_stream_t stream,
    g_device_ptr_t dst_buffer,
    size_t nbytes,
    const void* pattern,
    size_t pattern_nbytes
) {}

void execute_gpu_reduction_async(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    ReductionDescription reduction
) {}

void execute_gpu_fill_async(
    g_stream_t stream,
    g_device_ptr_t dst_buffer,
    const FillDescription& fill
) {}

}  // namespace kmm