#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "simplify_dims.hpp"

#include "kmm/core/const_value.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/copy.hpp"
#include "kmm/runtime/memops/reduction.hpp"

namespace kmm {

ReductionDescription ReductionDescription::simplify() const {
    ReductionDescription result = *this;

    result.num_dims = simplify_dims(
        dims,
        num_dims,
        result.dims,
        [](const ReductionDim& a, const ReductionDim& b) {
            return a.input_stride > b.input_stride;
        },
        [](ReductionDim& outer, const ReductionDim& inner) {
            if (outer.input_stride == inner.input_stride * inner.extent
                && outer.output_stride == inner.output_stride * inner.extent) {
                outer.extent *= inner.extent;
                outer.input_stride = inner.input_stride;
                outer.output_stride = inner.output_stride;
                return true;
            }

            return false;
        }
    );

    return result;
}

// Widens the half-open byte range [lo, hi) so that visiting `extent` positions spaced `stride`
// bytes apart (starting anywhere in the current range) stays inside it.
static void extend_range(
    ptrdiff_t& lo,
    ptrdiff_t& hi,
    memops_extent_type extent,
    memops_stride_type stride
) {
    ptrdiff_t span = checked_mul<ptrdiff_t>(extent - 1, stride);
    lo = span < 0 ? checked_add(lo, span) : lo;
    hi = span > 0 ? checked_add(hi, span) : hi;
}

// The byte range touched relative to `base_offset` when visiting every batch position (`dims`,
// via `stride_member`) and, for each, every reduced position (`reduced_extent`/`reduced_stride`).
// An axis of extent zero means nothing is visited, so the range collapses to empty.
static Range<ptrdiff_t> offset_range(
    ptrdiff_t base_offset,
    const ReductionDim* dims,
    size_t num_dims,
    size_t element_size,
    memops_stride_type ReductionDim::* stride_member,
    memops_extent_type reduced_extent,
    memops_stride_type reduced_stride
) {
    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i].extent < 1) {
            return {base_offset, base_offset};
        }
    }

    if (reduced_extent < 1) {
        return {base_offset, base_offset};
    }

    ptrdiff_t lo = base_offset;
    ptrdiff_t hi = base_offset;

    for (size_t i = 0; i < num_dims; i++) {
        extend_range(lo, hi, dims[i].extent, dims[i].*stride_member);
    }

    extend_range(lo, hi, reduced_extent, reduced_stride);

    return {lo, checked_add<ptrdiff_t>(hi, element_size)};
}

Range<ptrdiff_t> ReductionDescription::src_range() const {
    return offset_range(
        static_cast<ptrdiff_t>(input_offset),
        dims,
        num_dims,
        data_type_size(dtype),
        &ReductionDim::input_stride,
        reduction_extent,
        reduction_stride
    );
}

Range<ptrdiff_t> ReductionDescription::dst_range() const {
    return offset_range(
        static_cast<ptrdiff_t>(output_offset),
        dims,
        num_dims,
        data_type_size(dtype),
        &ReductionDim::output_stride,
        /* reduced_extent = */ 1,
        /* reduced_stride = */ 0
    );
}

CopyDescription ReductionDescription::as_copy() const {
    KMM_ASSERT(is_equivalent_to_copy());

    CopyDescription result(data_type_size(dtype));
    result.src_offset = input_offset;
    result.dst_offset = output_offset;
    result.num_dims = num_dims;

    for (size_t i = 0; i < num_dims; i++) {
        result.dims[i] = CopyDim {dims[i].extent, dims[i].input_stride, dims[i].output_stride};
    }

    return result;
}

/// Per-`ReductionOp` identity element and combining function, specialized per `T`/`Op` so that
/// `reduce_leaf` below has no runtime branch or function-pointer indirection to pick the operator.
template<typename T, ReductionOp Op>
struct ReduceTraits;

template<typename T>
struct ReduceTraits<T, ReductionOp::Sum> {
    static T identity() {
        return static_cast<T>(0);
    }

    static T combine(T a, T b) {
        return static_cast<T>(a + b);
    }
};

template<typename T>
struct ReduceTraits<T, ReductionOp::Product> {
    static T identity() {
        return static_cast<T>(1);
    }

    static T combine(T a, T b) {
        return static_cast<T>(a * b);
    }
};

template<typename T>
struct ReduceTraits<T, ReductionOp::Min> {
    static T identity() {
        return std::numeric_limits<T>::max();
    }

    static T combine(T a, T b) {
        return a < b ? a : b;
    }
};

template<typename T>
struct ReduceTraits<T, ReductionOp::Max> {
    static T identity() {
        return std::numeric_limits<T>::lowest();
    }

    static T combine(T a, T b) {
        return a > b ? a : b;
    }
};

template<typename T>
static FillValue reduction_identity_typed(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
            return FillValue::from(ReduceTraits<T, ReductionOp::Sum>::identity());
        case ReductionOp::Product:
            return FillValue::from(ReduceTraits<T, ReductionOp::Product>::identity());
        case ReductionOp::Min:
            return FillValue::from(ReduceTraits<T, ReductionOp::Min>::identity());
        case ReductionOp::Max:
            return FillValue::from(ReduceTraits<T, ReductionOp::Max>::identity());
    }

    KMM_PANIC("invalid reduction operator");
}

FillValue reduction_identity(DataType dtype, ReductionOp op) {
    switch (dtype) {
        case DataType::Int8:
            return reduction_identity_typed<int8_t>(op);
        case DataType::Int16:
            return reduction_identity_typed<int16_t>(op);
        case DataType::Int32:
            return reduction_identity_typed<int32_t>(op);
        case DataType::Int64:
            return reduction_identity_typed<int64_t>(op);
        case DataType::Uint8:
            return reduction_identity_typed<uint8_t>(op);
        case DataType::Uint16:
            return reduction_identity_typed<uint16_t>(op);
        case DataType::Uint32:
            return reduction_identity_typed<uint32_t>(op);
        case DataType::Uint64:
            return reduction_identity_typed<uint64_t>(op);
        case DataType::Float32:
            return reduction_identity_typed<float>(op);
        case DataType::Float64:
            return reduction_identity_typed<double>(op);
        case DataType::Unknown:
            break;
    }

    KMM_PANIC("invalid data type");
}

template<typename T, ReductionOp Op, bool Accumulate, typename E>
static void reduce_leaf(
    const std::byte* src,
    std::byte* dst,
    E reduction_extent,
    memops_stride_type reduction_stride
) {
    T acc = ReduceTraits<T, Op>::identity();

    for (memops_extent_type i = 0; i < reduction_extent; i++) {
        T value;
        std::memcpy(&value, src + i * reduction_stride, sizeof(T));
        acc = ReduceTraits<T, Op>::combine(acc, value);
    }

    if constexpr (Accumulate) {
        T previous;
        std::memcpy(&previous, dst, sizeof(T));
        acc = ReduceTraits<T, Op>::combine(previous, acc);
    }

    std::memcpy(dst, &acc, sizeof(T));
}

/// Recurses over the batch axes with `Rank` (the number of remaining axes) as a template
/// parameter, so the compiler can fully unroll the loop nest for the common, small ranks instead
/// of looping over a runtime-sized `dims` array.
template<typename T, ReductionOp Op, bool Accumulate, size_t Rank, typename E>
static void reduce_dim(
    const std::byte* src,
    std::byte* dst,
    const ReductionDim* dims,
    E reduction_extent,
    memops_stride_type reduction_stride
) {
    if constexpr (Rank == 0) {
        reduce_leaf<T, Op, Accumulate>(src, dst, reduction_extent, reduction_stride);
    } else {
        for (memops_extent_type i = 0; i < dims->extent; i++) {
            reduce_dim<T, Op, Accumulate, Rank - 1>(
                src + i * dims->input_stride,
                dst + i * dims->output_stride,
                dims + 1,
                reduction_extent,
                reduction_stride
            );
        }
    }
}

/// Dispatches the runtime `num_dims` (at most `MEMOPS_MAX_DIMS`, checked by the caller) to the
/// matching `reduce_dim<T, Op, Accumulate, Rank>` instantiation.
template<typename T, ReductionOp Op, bool Accumulate, size_t Rank = MEMOPS_MAX_DIMS, typename E>
static void reduce_dim_dispatch(
    size_t num_dims,
    const std::byte* src,
    std::byte* dst,
    const ReductionDim* dims,
    E reduction_extent,
    memops_stride_type reduction_stride
) {
    if (num_dims == Rank) {
        reduce_dim<T, Op, Accumulate, Rank>(src, dst, dims, reduction_extent, reduction_stride);
    } else if constexpr (Rank > 0) {
        reduce_dim_dispatch<T, Op, Accumulate, Rank - 1>(
            num_dims,
            src,
            dst,
            dims,
            reduction_extent,
            reduction_stride
        );
    } else {
        KMM_PANIC("invalid number of dimensions");
    }
}

template<typename T, ReductionOp Op, bool Accumulate>
static void reduce_op_accumulate(
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    // cases are:
    // * reduction_extent == 1: simply reduce src into dst
    // * reduction_extent == 2: reduce two buffers into one
    // * reduction_extent > 2: arbitrary reduction axis
    if (description.reduction_extent == 1) {
        reduce_dim_dispatch<T, Op, Accumulate>(
            description.num_dims,
            static_cast<const std::byte*>(src_addr) + description.input_offset,
            static_cast<std::byte*>(dst_addr) + description.output_offset,
            description.dims,
            ConstValue<memops_extent_type(1)>(),
            description.reduction_stride
        );
    } else if (description.reduction_extent == 2) {
        reduce_dim_dispatch<T, Op, Accumulate>(
            description.num_dims,
            static_cast<const std::byte*>(src_addr) + description.input_offset,
            static_cast<std::byte*>(dst_addr) + description.output_offset,
            description.dims,
            ConstValue<memops_extent_type(2)>(),
            description.reduction_stride
        );
    } else {
        reduce_dim_dispatch<T, Op, Accumulate>(
            description.num_dims,
            static_cast<const std::byte*>(src_addr) + description.input_offset,
            static_cast<std::byte*>(dst_addr) + description.output_offset,
            description.dims,
            description.reduction_extent,
            description.reduction_stride
        );
    }
}

template<typename T, ReductionOp Op>
static void reduce_op(
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    if (description.accumulate) {
        reduce_op_accumulate<T, Op, true>(src_addr, dst_addr, description);
    } else {
        reduce_op_accumulate<T, Op, false>(src_addr, dst_addr, description);
    }
}

template<typename T>
static void reduce_typed(
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    switch (description.operation) {
        case ReductionOp::Sum:
            return reduce_op<T, ReductionOp::Sum>(src_addr, dst_addr, description);
        case ReductionOp::Product:
            return reduce_op<T, ReductionOp::Product>(src_addr, dst_addr, description);
        case ReductionOp::Min:
            return reduce_op<T, ReductionOp::Min>(src_addr, dst_addr, description);
        case ReductionOp::Max:
            return reduce_op<T, ReductionOp::Max>(src_addr, dst_addr, description);
    }

    KMM_PANIC("invalid reduction operator");
}

void reduce(const void* src_addr, void* dst_addr, const ReductionDescription& description) {
    auto simplified = description.simplify();

    if (simplified.is_equivalent_to_copy()) {
        return copy(src_addr, dst_addr, simplified.as_copy());
    }

    switch (simplified.dtype) {
        case DataType::Unknown:
            break;
        case DataType::Int8:
            return reduce_typed<int8_t>(src_addr, dst_addr, simplified);
        case DataType::Int16:
            return reduce_typed<int16_t>(src_addr, dst_addr, simplified);
        case DataType::Int32:
            return reduce_typed<int32_t>(src_addr, dst_addr, simplified);
        case DataType::Int64:
            return reduce_typed<int64_t>(src_addr, dst_addr, simplified);
        case DataType::Uint8:
            return reduce_typed<uint8_t>(src_addr, dst_addr, simplified);
        case DataType::Uint16:
            return reduce_typed<uint16_t>(src_addr, dst_addr, simplified);
        case DataType::Uint32:
            return reduce_typed<uint32_t>(src_addr, dst_addr, simplified);
        case DataType::Uint64:
            return reduce_typed<uint64_t>(src_addr, dst_addr, simplified);
        case DataType::Float32:
            return reduce_typed<float>(src_addr, dst_addr, simplified);
        case DataType::Float64:
            return reduce_typed<double>(src_addr, dst_addr, simplified);
    }

    KMM_PANIC("invalid data type");
}

// `reduce_async` (the GPU counterpart) lives in reduction.cu -- it needs real GPU kernels (via
// CUB), so it must be compiled by nvcc, unlike the rest of this file.

}  // namespace kmm
