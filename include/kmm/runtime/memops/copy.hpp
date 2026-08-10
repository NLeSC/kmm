#pragma once

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/types.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/device_stream.hpp"

namespace kmm {

/// \addtogroup memops
/// @{

/// Describes a single axis of a strided copy: `extent` elements are copied, and each successive
/// element is offset by `src_stride`/`dst_stride` bytes in the source/destination buffer
/// respectively.
struct CopyDim {
    memops_extent_type extent = 1;
    memops_stride_type src_stride = 0;
    memops_stride_type dst_stride = 0;
};

/// Describes a (possibly strided, possibly multi-dimensional) copy from a source buffer to a
/// destination buffer.
///
/// The source and destination are treated as opaque byte buffers: `copy`/`copy_async` do not
/// interpret the bytes being copied, so `element_size` only determines the granularity of the
/// copy (i.e. how many contiguous bytes make up a single "element" along the innermost axis).
struct CopyDescription {
    /// The size (in bytes) of a single element.
    size_t element_size = 1;

    /// The number of axes described by `dims`. Must be at most `MEMOPS_MAX_DIMS`.
    size_t num_dims = 0;

    /// The extent and per-axis strides, ordered from the outermost to the innermost axis.
    CopyDim dims[MEMOPS_MAX_DIMS] = {};

    /// Appends an axis to this description. `src_stride`/`dst_stride` must be given in bytes.
    KMM_HOST_DEVICE
    void add_dimension(
        memops_extent_type extent,
        memops_stride_type src_stride,
        memops_stride_type dst_stride
    ) {
        KMM_ASSERT(num_dims < MEMOPS_MAX_DIMS);
        dims[num_dims] = CopyDim {extent, src_stride, dst_stride};
        num_dims++;
    }

    /// Returns the total number of elements copied (the product of the extent of each axis).
    KMM_HOST_DEVICE
    memops_extent_type num_elements() const {
        memops_extent_type result = 1;

        for (size_t i = 0; i < num_dims; i++) {
            result *= dims[i].extent;
        }

        return result;
    }

    /// Returns an equivalent description in canonical form. Axes are sorted by decreasing
    /// stride (i.e. largest stride is first) and
    CopyDescription simplify() const;
};

/// Copies data from `src_addr` to `dst_addr` on the CPU, according to `description`. Blocks
/// until the copy has completed.
void copy(const void* src_addr, void* dst_addr, const CopyDescription& description);

/// Copies data from `src_addr` to `dst_addr` on the GPU, according to `description`. The copy is
/// enqueued on `stream` after waiting for `dependencies`, and the returned event becomes ready
/// once the copy has completed. Either (or both) of `src_addr`/`dst_addr` may point to host or
/// device memory.
void copy_async(
    CUstream stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
);

/// @}

}  // namespace kmm
