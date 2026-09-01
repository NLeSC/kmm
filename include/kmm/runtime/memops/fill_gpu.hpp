#pragma once

#include "kmm/runtime/memops/fill.hpp"
#include "kmm/utils/backends.hpp"

namespace kmm::memops {

/// \addtogroup memops
/// @{

/// Fills `dst_addr` on the GPU, according to `description`, with copies of the fill value held in
/// `description.value` (the value bytes are read on the host before the work is enqueued). The
/// fill is enqueued asynchronously on `stream`.
void fill_gpu(g_stream_t stream, void* dst_addr, const FillDescription& description);

/// @}

}  // namespace kmm::memops
