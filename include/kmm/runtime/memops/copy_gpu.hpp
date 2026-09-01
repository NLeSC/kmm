#pragma once

#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream.hpp"
#include "kmm/runtime/memops/copy.hpp"

namespace kmm::memops {

/// \addtogroup memops
/// @{

/// Copies data from `src_addr` to `dst_addr` on the GPU, according to `description`. The copy is
/// enqueued on `stream` after waiting for `dependencies`, and the returned event becomes ready
/// once the copy has completed. Both `src_addr` and `dst_addr` must point to device memory on
/// the same device (i.e. this is a device-to-device copy).
void copy_gpu(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
);

/// @}

}  // namespace kmm::memops
