#pragma once

#include <cuda.h>

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/copy.hpp"
#include "kmm/runtime/memops/types.hpp"

namespace kmm {

/// \addtogroup memops
/// @{

/// Describes a single "batch" axis of a strided reduction: `extent` output elements are
/// produced, and each successive output element is offset by `input_stride`/`output_stride`
/// bytes in the input/output buffer respectively.
///
/// This describes the axes that are *not* reduced over; see `ReductionDescription::reduction_extent`
/// for the axis that is reduced over.
struct ReductionDim {
    memops_extent_type extent = 1;
    memops_stride_type input_stride = 0;
    memops_stride_type output_stride = 0;
};

/// Describes a (possibly strided, possibly multi-dimensional) reduction of an input buffer into
/// an output buffer.
///
/// A reduction combines groups of input elements into a single output element using `operation`
/// (e.g. `Sum`). `dims` describes the "batch" axes: one output element is produced per
/// combination of batch indices. `reduction_extent`/`reduction_stride` describe the additional
/// axis that is reduced over: for each output element, `reduction_extent` input elements
/// (spaced `reduction_stride` bytes apart, starting at the corresponding batch position) are
/// combined together.
///
/// For example, reducing an `(m, n)` row-major input down to `m` outputs (i.e. reducing over the
/// trailing axis) is described by a single batch dimension with `extent = m` and
/// `input_stride = n * element_size`, together with `reduction_extent = n` and
/// `reduction_stride = element_size`.
struct ReductionDescription {
    /// The element type operated on. Determines both how elements are interpreted numerically
    /// and the size (in bytes) of a single element.
    DataType dtype = DataType::Float32;

    /// The operator used to combine elements.
    ReductionOp operation = ReductionOp::Sum;

    /// A byte offset added to `src_addr` before applying `dims`/`reduction_stride`.
    memops_stride_type input_offset = 0;

    /// A byte offset added to `dst_addr` before applying `dims`.
    memops_stride_type output_offset = 0;

    /// The number of batch axes described by `dims`. Must be at most `MEMOPS_MAX_DIMS`.
    size_t num_dims = 0;

    /// The extent and per-axis strides of the batch axes, ordered from the outermost to the
    /// innermost axis.
    ReductionDim dims[MEMOPS_MAX_DIMS] = {};

    /// The number of input elements combined into each output element.
    memops_extent_type reduction_extent = 1;

    /// The byte offset between successive input elements being combined into the same output
    /// element.
    memops_stride_type reduction_stride = 0;

    /// If `true`, each output element is combined with its previous value (i.e. the output
    /// buffer is treated as already holding a partial result). If `false`, each output element
    /// is overwritten with the result of the reduction.
    bool accumulate = false;

    /// Appends a batch axis to this description. `input_stride`/`output_stride` must be given in
    /// bytes.
    KMM_HOST_DEVICE
    void add_dimension(
        memops_extent_type extent,
        memops_stride_type input_stride,
        memops_stride_type output_stride
    ) {
        KMM_ASSERT(num_dims < MEMOPS_MAX_DIMS);
        dims[num_dims] = ReductionDim {extent, input_stride, output_stride};
        num_dims++;
    }

    /// Returns the total number of output elements produced (the product of the extent of each
    /// batch axis).
    KMM_HOST_DEVICE
    memops_extent_type num_outputs() const {
        memops_extent_type result = 1;

        for (size_t i = 0; i < num_dims; i++) {
            result *= dims[i].extent;
        }

        return result;
    }

    /// Returns a copy of this description with `dims` sorted from the largest `input_stride` to
    /// the smallest, and adjacent axes merged whenever they are contiguous (i.e. the outer axis's
    /// `input_stride`/`output_stride` equal the inner axis's `extent * input_stride`/
    /// `extent * output_stride`). Does not touch `reduction_extent`/`reduction_stride`, since
    /// those describe a different axis (the one reduced over, shared by every output element).
    ReductionDescription simplify() const;

    /// Returns `true` if this reduction combines exactly one input element into each output
    /// element without accumulating into the previous value: every `combine(identity, value)`
    /// then reduces to `value` (for all of `Sum`/`Product`/`Min`/`Max`), so the reduction is
    /// equivalent to a plain strided copy from `src_addr` to `dst_addr`. See `as_copy`.
    KMM_HOST_DEVICE
    bool is_equivalent_to_copy() const {
        return reduction_extent == 1 && !accumulate;
    }

    /// Converts this reduction into an equivalent `CopyDescription`. Only valid when
    /// `is_equivalent_to_copy()` returns `true`.
    CopyDescription as_copy() const;
};

/// Reduces `src_addr` into `dst_addr` on the CPU, according to `description`. Blocks until the
/// reduction has completed.
void reduce(const void* src_addr, void* dst_addr, const ReductionDescription& description);

/// Reduces `src_addr` into `dst_addr` on the GPU, according to `description`. The reduction is
/// enqueued on `stream` after waiting for `dependencies`, and the returned event becomes ready
/// once the reduction has completed.
void reduce_async(
    CUstream stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
);

/// @}

}  // namespace kmm
