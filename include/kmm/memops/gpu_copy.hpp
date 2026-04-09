#pragma once

#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

void execute_gpu_h2d_copy_async(
    g_stream_t stream,
    const void* src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
);

void execute_gpu_d2h_copy_async(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    void* dst_buffer,
    CopyDef copy_description
);

void execute_gpu_d2d_copy_async(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
);

void execute_gpu_h2d_copy(
    const void* src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
);

void execute_gpu_d2h_copy(g_device_ptr_t src_buffer, void* dst_buffer, CopyDef copy_description);

void execute_gpu_d2d_copy(
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    CopyDef copy_description
);

}  // namespace kmm