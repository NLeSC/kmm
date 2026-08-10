#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "simplify_dims.hpp"

namespace kmm {

ReductionDescription ReductionDescription::simplify() const {
    ReductionDescription result = *this;

    result.num_dims = simplify_dims(
        dims,
        num_dims,
        result.dims,
        [](const ReductionDim& a, const ReductionDim& b) { return a.input_stride > b.input_stride; },
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

template<typename T>
static void reduce_leaf(
    const std::byte* src,
    std::byte* dst,
    memops_extent_type reduction_extent,
    memops_stride_type reduction_stride,
    T identity,
    T (*combine)(T, T),
    bool accumulate
) {
    T acc = identity;

    for (memops_extent_type i = 0; i < reduction_extent; i++) {
        T value;
        std::memcpy(&value, src + i * reduction_stride, sizeof(T));
        acc = combine(acc, value);
    }

    if (accumulate) {
        T previous;
        std::memcpy(&previous, dst, sizeof(T));
        acc = combine(previous, acc);
    }

    std::memcpy(dst, &acc, sizeof(T));
}

template<typename T>
static void reduce_dim(
    const std::byte* src,
    std::byte* dst,
    const ReductionDim* dims,
    size_t num_dims,
    memops_extent_type reduction_extent,
    memops_stride_type reduction_stride,
    T identity,
    T (*combine)(T, T),
    bool accumulate
) {
    if (num_dims == 0) {
        reduce_leaf<T>(src, dst, reduction_extent, reduction_stride, identity, combine, accumulate);
        return;
    }

    for (memops_extent_type i = 0; i < dims->extent; i++) {
        reduce_dim<T>(
            src + i * dims->input_stride,
            dst + i * dims->output_stride,
            dims + 1,
            num_dims - 1,
            reduction_extent,
            reduction_stride,
            identity,
            combine,
            accumulate
        );
    }
}

template<typename T>
static void reduce_typed(
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    T identity;
    T (*combine)(T, T);

    switch (description.operation) {
        case ReductionOp::Sum:
            identity = static_cast<T>(0);
            combine = [](T a, T b) -> T { return static_cast<T>(a + b); };
            break;
        case ReductionOp::Product:
            identity = static_cast<T>(1);
            combine = [](T a, T b) -> T { return static_cast<T>(a * b); };
            break;
        case ReductionOp::Min:
            identity = std::numeric_limits<T>::max();
            combine = [](T a, T b) -> T { return a < b ? a : b; };
            break;
        case ReductionOp::Max:
            identity = std::numeric_limits<T>::lowest();
            combine = [](T a, T b) -> T { return a > b ? a : b; };
            break;
        default:
            KMM_PANIC("invalid reduction operator");
    }

    reduce_dim<T>(
        static_cast<const std::byte*>(src_addr),
        static_cast<std::byte*>(dst_addr),
        description.dims,
        description.num_dims,
        description.reduction_extent,
        description.reduction_stride,
        identity,
        combine,
        description.accumulate
    );
}

void reduce(const void* src_addr, void* dst_addr, const ReductionDescription& description) {
    switch (description.dtype) {
        case DataType::Unknown:
            break;
        case DataType::Int8:
            return reduce_typed<int8_t>(src_addr, dst_addr, description);
        case DataType::Int16:
            return reduce_typed<int16_t>(src_addr, dst_addr, description);
        case DataType::Int32:
            return reduce_typed<int32_t>(src_addr, dst_addr, description);
        case DataType::Int64:
            return reduce_typed<int64_t>(src_addr, dst_addr, description);
        case DataType::Uint8:
            return reduce_typed<uint8_t>(src_addr, dst_addr, description);
        case DataType::Uint16:
            return reduce_typed<uint16_t>(src_addr, dst_addr, description);
        case DataType::Uint32:
            return reduce_typed<uint32_t>(src_addr, dst_addr, description);
        case DataType::Uint64:
            return reduce_typed<uint64_t>(src_addr, dst_addr, description);
        case DataType::Float32:
            return reduce_typed<float>(src_addr, dst_addr, description);
        case DataType::Float64:
            return reduce_typed<double>(src_addr, dst_addr, description);
    }

    KMM_PANIC("invalid data type");
}

// `reduce_async` (the GPU counterpart) lives in reduction.cu -- it needs real GPU kernels (via
// CUB), so it must be compiled by nvcc, unlike the rest of this file.

}  // namespace kmm
