#pragma once

#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

void execute_gpu_fill_async(g_stream_t stream, g_device_ptr_t dst_buffer, const FillDef& fill);

}