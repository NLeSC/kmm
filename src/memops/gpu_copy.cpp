
#include <stdexcept>

#include "fmt/format.h"

#include "kmm/memops/types.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

void throw_unsupported_dimension_exception(size_t dim) {
    throw std::runtime_error(
        fmt::format(
            "copy operation is {} dimensional, only 1D or 2D copy operations are supported",
            dim + 1
        )
    );
}

void execute_gpu_h2d_copy_impl(
    std::optional<g_stream_t> stream,
    const void* src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    g_device_ptr_t dst_ptr = gpu_deviceptr_offset(dst_buffer, copy_description.dst_offset);
    const void* src_ptr = static_cast<const uint8_t*>(src_buffer) + copy_description.src_offset;

    if (dim == 0) {
        if (stream) {
            KMM_GPU_CHECK(g_memcpy_h_to_d_async(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size,
                *stream
            ));
        } else {
            KMM_GPU_CHECK(g_memcpy_h_to_d(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size
            ));
        }
    } else if (dim == 1) {
        gpu_memcpy2d_t info;
        ::memset(&info, 0, sizeof(gpu_memcpy2d_t));

        info.srcMemoryType = g_memory_type_t::G_MEMORYTYPE_HOST;
        info.srcHost = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = g_memory_type_t::G_MEMORYTYPE_DEVICE;
        info.dstDevice = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_GPU_CHECK(g_memcpy_2d_async(&info, *stream));
        } else {
            KMM_GPU_CHECK(g_memcpy_2d(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_gpu_d2h_copy_impl(
    std::optional<g_stream_t> stream,
    g_device_ptr_t src_buffer,
    void* dst_buffer,
    CopyDef copy_description
) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    void* dst_ptr = static_cast<uint8_t*>(dst_buffer) + copy_description.dst_offset;
    g_device_ptr_t src_ptr = gpu_deviceptr_offset(src_buffer, copy_description.src_offset);

    if (dim == 0) {
        if (stream) {
            KMM_GPU_CHECK(
                g_memcpy_d_to_h_async(dst_ptr, src_ptr, copy_description.element_size, *stream)
            );
        } else {
            KMM_GPU_CHECK(g_memcpy_d_to_h(dst_ptr, src_ptr, copy_description.element_size));
        }
    } else if (dim == 1) {
        gpu_memcpy2d_t info;
        ::memset(&info, 0, sizeof(gpu_memcpy2d_t));

        info.srcMemoryType = g_memory_type_t::G_MEMORYTYPE_DEVICE;
        info.srcDevice = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = g_memory_type_t::G_MEMORYTYPE_HOST;
        info.dstHost = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_GPU_CHECK(g_memcpy_2d_async(&info, *stream));
        } else {
            KMM_GPU_CHECK(g_memcpy_2d(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_gpu_d2d_copy_impl(
    std::optional<g_stream_t> stream,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    g_device_ptr_t dst_ptr = gpu_deviceptr_offset(dst_buffer, copy_description.dst_offset);
    g_device_ptr_t src_ptr = gpu_deviceptr_offset(src_buffer, copy_description.src_offset);

    if (dim == 0) {
        if (stream) {
            KMM_GPU_CHECK(g_memcpy_d_to_d_async(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size,
                *stream
            ));
        } else {
            KMM_GPU_CHECK(g_memcpy_d_to_d(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size
            ));
        }
    } else if (dim == 1) {
        gpu_memcpy2d_t info;
        ::memset(&info, 0, sizeof(gpu_memcpy2d_t));

        info.srcMemoryType = g_memory_type_t::G_MEMORYTYPE_DEVICE;
        info.srcDevice = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = g_memory_type_t::G_MEMORYTYPE_DEVICE;
        info.dstDevice = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_GPU_CHECK(g_memcpy_2d_async(&info, *stream));
        } else {
            KMM_GPU_CHECK(g_memcpy_2d(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_gpu_h2d_copy(
    const void* src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
) {
    execute_gpu_h2d_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_h2d_copy_async(
    g_stream_t stream,
    const void* src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
) {
    execute_gpu_h2d_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2h_copy(g_device_ptr_t src_buffer, void* dst_buffer, CopyDef copy_description) {
    execute_gpu_d2h_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2h_copy_async(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    void* dst_buffer,
    CopyDef copy_description
) {
    execute_gpu_d2h_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2d_copy(
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
) {
    execute_gpu_d2d_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2d_copy_async(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
) {
    execute_gpu_d2d_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}
}  // namespace kmm