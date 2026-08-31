#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/checked_math.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/core/range.hpp"
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
struct CopyDescription {
    /// The size (in bytes) of a single element.
    size_t element_size = 1;

    /// A byte offset added to `src_addr` before applying `dims`.
    memops_stride_type src_offset = 0;

    /// A byte offset added to `dst_addr` before applying `dims`.
    memops_stride_type dst_offset = 0;

    /// The number of axes described by `dims`. Must be at most `MEMOPS_MAX_DIMS`.
    size_t num_dims = 0;

    /// The extent and per-axis strides, ordered from the outermost to the innermost axis.
    CopyDim dims[MEMOPS_MAX_DIMS] = {};

    CopyDescription() = default;

    KMM_HOST_DEVICE
    explicit CopyDescription(size_t element_size) : element_size(element_size) {}

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

    /// Returns the half-open range of byte offsets (relative to `src_addr`) that this copy
    /// will read from.
    Range<ptrdiff_t> src_range() const;

    /// Returns the half-open range of byte offsets (relative to `dst_addr`) that this copy
    /// will write to.
    Range<ptrdiff_t> dst_range() const;

    /// Returns an equivalent description in canonical form. Axes are sorted by decreasing
    /// stride (i.e. largest stride is first) and
    CopyDescription simplify() const;
};

/// Copies data from `src_addr` to `dst_addr` on the CPU, according to `description`. Blocks
/// until the copy has completed.
void copy(const void* src_addr, void* dst_addr, const CopyDescription& description);

/// Copies data from `src_addr` to `dst_addr` on the GPU, according to `description`. The copy is
/// enqueued on `stream` after waiting for `dependencies`, and the returned event becomes ready
/// once the copy has completed. Both `src_addr` and `dst_addr` must point to device memory on
/// the same device (i.e. this is a device-to-device copy).
void copy_async(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
);

/// Builds a `CopyDescription` that copies `element_size`-sized elements between two layouts of
/// matching rank (e.g. `kmm::Layout`), translating each layout's (element-space) base offset,
/// per-axis origin, and strides into the byte offsets/strides that `CopyDescription` expects.
template<typename DstLayoutT, typename SrcLayoutT>
CopyDescription make_copy_description(
    const DstLayoutT& dst,
    const SrcLayoutT& src,
    size_t element_size
) {
    static_assert(DstLayoutT::rank == SrcLayoutT::rank, "rank mismatch");
    static_assert(DstLayoutT::rank <= MEMOPS_MAX_DIMS, "rank exceeds maximum");

    ptrdiff_t dst_offset = dst.base_offset();
    ptrdiff_t src_offset = src.base_offset();

    for (size_t i = 0; i < DstLayoutT::rank; i++) {
        dst_offset += static_cast<ptrdiff_t>(dst.stride(i)) * static_cast<ptrdiff_t>(dst.begin(i));
        src_offset += static_cast<ptrdiff_t>(src.stride(i)) * static_cast<ptrdiff_t>(src.begin(i));
    }

    CopyDescription descr(element_size);
    descr.src_offset = checked_mul<memops_stride_type>(src_offset, element_size);
    descr.dst_offset = checked_mul<memops_stride_type>(dst_offset, element_size);

    for (size_t i = 0; i < DstLayoutT::rank; i++) {
        descr.add_dimension(
            checked_cast<memops_extent_type>(dst.extent(i)),
            checked_mul<memops_stride_type>(src.stride(i), element_size),
            checked_mul<memops_stride_type>(dst.stride(i), element_size)
        );
    }

    return descr.simplify();
}

/// @}

}  // namespace kmm
