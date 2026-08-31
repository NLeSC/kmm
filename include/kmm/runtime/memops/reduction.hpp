#pragma once

#include "kmm/core/checked_math.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/copy.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memops/types.hpp"

namespace kmm {

/// The identity element for `op` at `dtype`, as the raw bytes to broadcast-fill a buffer that is
/// about to be reduced into (so slots never written still drop out of the fold): `0` for `Sum`,
/// `1` for `Product`, the type's maximum for `Min`, its lowest for `Max`.
FillValue reduction_identity(DataType dtype, ReductionOp op);

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

    ReductionDescription() = default;

    KMM_HOST_DEVICE
    ReductionDescription(DataType dtype, ReductionOp operation) :
        dtype(dtype),
        operation(operation) {}

    /// Appends a batch axis to this description. `input_stride`/`output_stride` must be given in
    /// bytes.
    ///
    /// An axis of extent one is dropped (it is visited exactly once, so its stride never
    /// contributes to addressing). Otherwise, this axis is checked against every existing axis
    /// for contiguity (in both `input_stride` and `output_stride`) and folded into the first one
    /// it is contiguous with instead of consuming a new slot. This keeps `num_dims` from growing
    /// unnecessarily, which matters since `dims` only has room for `MEMOPS_MAX_DIMS` axes.
    KMM_HOST_DEVICE
    void add_dimension(
        memops_extent_type extent,
        memops_stride_type input_stride,
        memops_stride_type output_stride
    ) {
        if (extent == 1) {
            return;
        }

        for (size_t i = 0; i < num_dims; i++) {
            ReductionDim& dim = dims[i];

            // `dim` is the outer neighbor of the new axis: it keeps its extent, but adopts the
            // new axis's (smaller) stride as its own.
            if (dim.input_stride == input_stride * extent
                && dim.output_stride == output_stride * extent) {
                dim.extent *= extent;
                dim.input_stride = input_stride;
                dim.output_stride = output_stride;
                return;
            }

            // `dim` is the inner neighbor of the new axis: the new axis's stride already
            // matches `dim`'s span, so only `dim`'s extent needs to grow.
            if (input_stride == dim.input_stride * dim.extent
                && output_stride == dim.output_stride * dim.extent) {
                dim.extent *= extent;
                return;
            }
        }

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

    /// Returns the half-open range of byte offsets (relative to `src_addr`) that this reduction
    /// reads from. Accounts for `input_offset`, the batch axes (`dims`), and the reduced axis
    /// (`reduction_extent`/`reduction_stride`).
    Range<ptrdiff_t> src_range() const;

    /// Returns the half-open range of byte offsets (relative to `dst_addr`) that this reduction
    /// writes to. Accounts for `output_offset` and the batch axes (`dims`).
    Range<ptrdiff_t> dst_range() const;

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

    /// Returns `true` if this reduction combines *zero* input elements into each output element
    /// (`reduction_extent == 0`) without accumulating into the previous value: every output is
    /// therefore left as the identity element for `operation`, so the reduction is equivalent to
    /// broadcasting that identity across the output region. See `as_fill`.
    KMM_HOST_DEVICE
    bool is_equivalent_to_fill() const {
        return reduction_extent == 0 && !accumulate;
    }

    /// Converts this reduction into an equivalent `FillDescription` that writes the identity
    /// element of `operation` across the output region (the batch axes). Only valid when
    /// `is_equivalent_to_fill()` returns `true`.
    FillDescription as_fill() const;

    /// Returns `true` if this reduction is guaranteed to leave the output buffer unchanged, so it
    /// need not read or write anything. That happens when no output elements are produced at all
    /// (`num_outputs() == 0`), or when zero input elements are combined into each output
    /// (`reduction_extent == 0`) while accumulating: every output becomes
    /// `combine(previous, identity) == previous` for all of `Sum`/`Product`/`Min`/`Max`.
    KMM_HOST_DEVICE
    bool is_noop() const {
        return num_outputs() == 0 || (reduction_extent == 0 && accumulate);
    }
};

/// Builds a `ReductionDescription` that reduces `src` (e.g. a `kmm::Layout`) over `axis` into
/// `dst`, whose rank must be exactly one less than `src`'s: every axis of `src` other than `axis`
/// maps (in order) to the corresponding axis of `dst`, and `axis` itself becomes the reduced
/// axis. Translates each layout's (element-space) base offset, per-axis origin, and strides into
/// the byte offsets/strides that `ReductionDescription` expects.
template<typename DstLayoutT, typename SrcLayoutT>
ReductionDescription make_reduction_description(
    const DstLayoutT& dst,
    const SrcLayoutT& src,
    size_t axis,
    DataType dtype,
    ReductionOp op
) {
    static_assert(
        DstLayoutT::rank + 1 == SrcLayoutT::rank,
        "dst must have exactly one axis fewer than src"
    );
    static_assert(SrcLayoutT::rank <= MEMOPS_MAX_DIMS + 1, "rank exceeds maximum");
    KMM_ASSERT(axis < SrcLayoutT::rank);

    size_t element_size = data_type_size(dtype);

    ptrdiff_t dst_offset = dst.base_offset();
    ptrdiff_t src_offset = src.base_offset();

    for (size_t i = 0; i < DstLayoutT::rank; i++) {
        dst_offset += static_cast<ptrdiff_t>(dst.stride(i)) * static_cast<ptrdiff_t>(dst.begin(i));
    }

    for (size_t i = 0; i < SrcLayoutT::rank; i++) {
        src_offset += static_cast<ptrdiff_t>(src.stride(i)) * static_cast<ptrdiff_t>(src.begin(i));
    }

    ReductionDescription descr(dtype, op);
    descr.input_offset = checked_mul<memops_stride_type>(src_offset, element_size);
    descr.output_offset = checked_mul<memops_stride_type>(dst_offset, element_size);
    descr.reduction_extent = checked_cast<memops_extent_type>(src.extent(axis));
    descr.reduction_stride = checked_mul<memops_stride_type>(src.stride(axis), element_size);

    size_t dst_axis = 0;

    for (size_t i = 0; i < SrcLayoutT::rank; i++) {
        if (i == axis) {
            continue;
        }

        descr.add_dimension(
            checked_cast<memops_extent_type>(dst.extent(dst_axis)),
            checked_mul<memops_stride_type>(src.stride(i), element_size),
            checked_mul<memops_stride_type>(dst.stride(dst_axis), element_size)
        );

        dst_axis++;
    }

    return descr.simplify();
}

/// @}

namespace memops {

/// \addtogroup memops
/// @{

/// Reduces `src_addr` into `dst_addr` on the CPU, according to `description`. Blocks until the
/// reduction has completed.
void reduce(const void* src_addr, void* dst_addr, const ReductionDescription& description);

/// @}

}  // namespace memops

}  // namespace kmm
