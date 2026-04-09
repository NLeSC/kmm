#pragma once

#include "kmm/core/backends.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

/**
 *
 */
void execute_gpu_reduction_async(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    ReductionDef reduction
);

}  // namespace kmm