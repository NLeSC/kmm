#pragma once

#include "kmm/runtime/memops/copy_gpu.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/utils/backends.hpp"

namespace kmm::memops {

/// \addtogroup memops
/// @{

/// Reduces `src_addr` into `dst_addr` on the GPU, according to `description`. The reduction is
/// enqueued on `stream` after waiting for `dependencies`, and the returned event becomes ready
/// once the reduction has completed.
void reduce_gpu(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    void* scratch_addr,
    const ReductionDescription& description
);

/// Returns the size in bytes of the scratch buffer that `reduce_gpu` requires for `description`.
size_t reduce_gpu_scratch_size(const ReductionDescription& description);

/// @}

}  // namespace kmm::memops
