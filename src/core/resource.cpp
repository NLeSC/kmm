#include <stdexcept>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/core/resource.hpp"
#include "kmm/memops/gpu_fill.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

InvalidResourceException::InvalidResourceException(
    const std::type_info& expected,
    const std::type_info& gotten
) {
    m_message = fmt::format(
        "task expected an execution context of type {}, but was executed with type {}",
        expected.name(),
        gotten.name()
    );
}

const char* InvalidResourceException::what() const noexcept {
    return m_message.c_str();
}

DeviceResource::DeviceResource(DeviceInfo info, GPUContextHandle context, g_stream_t stream) :
    DeviceInfo(info),
    m_context(context),
    m_stream(stream) {
    GPUContextGuard guard {m_context};

    KMM_GPU_CHECK(blas_create(&m_blas_handle));
    KMM_GPU_CHECK(blas_set_stream(m_blas_handle, m_stream));
}

DeviceResource::~DeviceResource() {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(blas_destroy(m_blas_handle));
}

void DeviceResource::synchronize() const {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_stream_synchronize(nullptr));
    KMM_GPU_CHECK(g_stream_synchronize(m_stream));
}

void DeviceResource::fill_bytes(
    void* dest_buffer,
    size_t nbytes,
    const void* fill_pattern,
    size_t fill_pattern_size
) const {
    GPUContextGuard guard {m_context};
    execute_gpu_fill_async(
        m_stream,
        reinterpret_cast<g_device_ptr_t>(dest_buffer),
        FillDef(fill_pattern_size, nbytes / fill_pattern_size, fill_pattern)
    );
}

void DeviceResource::copy_bytes(const void* source_buffer, void* dest_buffer, size_t nbytes) const {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(g_memcpy_async(
        reinterpret_cast<g_device_ptr_t>(dest_buffer),
        reinterpret_cast<g_device_ptr_t>(const_cast<void*>(source_buffer)),
        nbytes,
        m_stream
    ));
}

}  // namespace kmm