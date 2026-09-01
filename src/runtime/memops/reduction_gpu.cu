#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cub/block/block_reduce.cuh"
#include "reduction_traits.hpp"

#include "kmm/core/const_value.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/fill_gpu.hpp"
#include "kmm/runtime/memops/reduction_gpu.hpp"

namespace kmm::memops {

template<typename T, ReductionOp Op, size_t Rank>
__global__ void elementwise_reduce_kernel(
    const void* src_addr,
    void* dst_addr,
    Vec<ReductionDim, Rank> dims,
    memops_extent_type reduction_extent,
    memops_stride_type reduction_stride,
    bool accumulate
) {
    uint p[3] = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    };

    for (size_t i = 0; i < Rank; i++) {
        if (p[i] >= dims[i].extent) {
            return;
        }

        src_addr = static_cast<const std::byte*>(src_addr) + dims[i].input_stride * p[i];
        dst_addr = static_cast<std::byte*>(dst_addr) + dims[i].output_stride * p[i];
    }

    T result = ReductionTraits<T, Op>::identity();

    for (memops_extent_type i = 0; i < reduction_extent; i++) {
        result = ReductionTraits<T, Op>::combine(result, *static_cast<const T*>(src_addr));
        src_addr = static_cast<const std::byte*>(src_addr) + reduction_stride;
    }

    if (accumulate) {
        result = ReductionTraits<T, Op>::combine(result, *static_cast<const T*>(dst_addr));
    }

    *static_cast<T*>(dst_addr) = result;
}

template<typename T, ReductionOp Op>
void launch_elementwise_reduce(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    Vec<ReductionDim, 3> dims;

    for (size_t i = 0; i < 3; i++) {
        if (i < description.num_dims) {
            dims[i] = description.dims[i];
        } else {
            // default is extent=1, stride=0
            dims[i] = ReductionDim {};
        }
    }

    dim3 block_size = {256, 1, 1};
    while (is_less(description.dims[0].extent, block_size.x)) {
        block_size.x /= 2;
        block_size.y *= 2;
    }

    while (is_less(description.dims[1].extent, block_size.y)) {
        block_size.y /= 2;
        block_size.z *= 2;
    }

    dim3 grid_size = {
        div_ceil(checked_cast<uint>(description.dims[0].extent), block_size.x),
        div_ceil(checked_cast<uint>(description.dims[1].extent), block_size.y),
        div_ceil(checked_cast<uint>(description.dims[2].extent), block_size.z),
    };

    elementwise_reduce_kernel<T, Op><<<grid_size, block_size, 0, stream>>>(
        static_cast<const std::byte*>(src_addr) + description.input_offset,
        static_cast<std::byte*>(dst_addr) + description.output_offset,
        dims,
        description.reduction_extent,
        description.reduction_stride,
        description.accumulate
    );
}

template<uint BlockDim, typename T, ReductionOp Op, size_t Rank>
__global__ void blockwise_reduce_kernel(
    const void* src_addr,
    void* dst_addr,
    Vec<ReductionDim, Rank> dims,
    memops_extent_type reduction_extent,
    memops_stride_type reduction_stride,
    bool accumulate
) {
    __shared__ typename cub::BlockReduce<T, BlockDim>::TempStorage smem_storage;
    cub::BlockReduce<T, BlockDim> block_reduce {smem_storage};

    uint p[3] = {blockIdx.x, blockIdx.y, blockIdx.z};

    for (size_t i = 0; i < Rank; i++) {
        if (p[i] >= dims[i].extent) {
            return;
        }

        src_addr = static_cast<const std::byte*>(src_addr) + dims[i].input_stride * p[i];
        dst_addr = static_cast<std::byte*>(dst_addr) + dims[i].output_stride * p[i];
    }

    T result = ReductionTraits<T, Op>::identity();

    for (memops_extent_type i = threadIdx.x; i < reduction_extent; i += BlockDim) {
        result = ReductionTraits<T, Op>::combine(result, *static_cast<const T*>(src_addr));
        src_addr = static_cast<const std::byte*>(src_addr) + reduction_stride;
    }

    result = block_reduce.Reduce(result, ReductionTraits<T, Op>::combine);

    if (threadIdx.x == 0) {
        if (accumulate) {
            result = ReductionTraits<T, Op>::combine(result, *static_cast<const T*>(dst_addr));
        }

        *static_cast<T*>(dst_addr) = result;
    }
}

template<typename T, ReductionOp Op>
void launch_blockwise_reduce(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    static constexpr uint block_size = 256;

    Vec<ReductionDim, 3> dims;

    for (size_t i = 0; i < 3; i++) {
        if (i < description.num_dims) {
            dims[i] = description.dims[i];
        } else {
            // default is extent=1, stride=0
            dims[i] = ReductionDim {};
        }
    }

    dim3 grid_size = {
        checked_cast<uint>(description.dims[0].extent),
        checked_cast<uint>(description.dims[1].extent),
        checked_cast<uint>(description.dims[2].extent),
    };

    blockwise_reduce_kernel<block_size, T, Op><<<grid_size, block_size, 0, stream>>>(
        static_cast<const std::byte*>(src_addr) + description.input_offset,
        static_cast<std::byte*>(dst_addr) + description.output_offset,
        dims,
        description.reduction_extent,
        description.reduction_stride,
        description.accumulate
    );
}

template<typename T, ReductionOp Op>
void reduced_typed_op_async(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    if (description.reduction_extent < 1024) {
        return launch_elementwise_reduce<T, Op>(stream, src_addr, dst_addr, description);
    } else {
        return launch_blockwise_reduce<T, Op>(stream, src_addr, dst_addr, description);
    }
}

/// Runs `reduced_typed_op_async<T, Op>` when that combination is a supported reduction (see
/// `is_reduction_supported`), otherwise throws instead of failing to compile. This is what lets
/// `reduce_typed_async` list every operator unconditionally even for element types that only
/// support a subset (integers-only for the bitwise ops, `KeyValue` only for `Min`/`Max`).
template<typename T, ReductionOp Op>
void reduced_typed_op_checked_async(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    if constexpr (is_reduction_supported<T, Op>) {
        reduced_typed_op_async<T, Op>(stream, src_addr, dst_addr, description);
    } else {
        throw std::runtime_error("reduction operator not supported for this data type");
    }
}

template<typename T>
void reduce_typed_async(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    switch (description.operation) {
        case ReductionOp::Sum:
            return reduced_typed_op_checked_async<T, ReductionOp::Sum>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::Product:
            return reduced_typed_op_checked_async<T, ReductionOp::Product>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::Min:
            return reduced_typed_op_checked_async<T, ReductionOp::Min>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::Max:
            return reduced_typed_op_checked_async<T, ReductionOp::Max>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::BitwiseAnd:
            return reduced_typed_op_checked_async<T, ReductionOp::BitwiseAnd>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::BitwiseOr:
            return reduced_typed_op_checked_async<T, ReductionOp::BitwiseOr>(
                stream,
                src_addr,
                dst_addr,
                description
            );
    }

    KMM_PANIC("invalid reduction operator");
}

void do_reduce_gpu(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    switch (description.dtype) {
        case DataType::Unknown:
            break;
        case DataType::Int32:
            return reduce_typed_async<int32_t>(stream, src_addr, dst_addr, description);
        case DataType::Int64:
            return reduce_typed_async<int64_t>(stream, src_addr, dst_addr, description);
        case DataType::Uint32:
            return reduce_typed_async<uint32_t>(stream, src_addr, dst_addr, description);
        case DataType::Uint64:
            return reduce_typed_async<uint64_t>(stream, src_addr, dst_addr, description);
        case DataType::Float32:
            return reduce_typed_async<float>(stream, src_addr, dst_addr, description);
        case DataType::Float64:
            return reduce_typed_async<double>(stream, src_addr, dst_addr, description);
        case DataType::KeyValueInt64:
            return reduce_typed_async<KeyValue<int64_t>>(stream, src_addr, dst_addr, description);
        case DataType::KeyValueFloat64:
            return reduce_typed_async<KeyValue<double>>(stream, src_addr, dst_addr, description);
    }

    KMM_PANIC("invalid data type");
}

void do_multilevel_reduce(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    memops_extent_type elements_per_block = 1024;
    auto num_chunks = div_floor(description.reduction_extent, elements_per_block);
    auto chunk_stride = checked_mul(description.reduction_stride, elements_per_block);
    auto remaining_chunks = description.reduction_extent % elements_per_block;

    ReductionDescription coarse(description.dtype, description.operation);
    coarse.input_offset = description.input_offset;
    coarse.output_offset = 0;
    coarse.reduction_extent = elements_per_block;
    coarse.reduction_stride = description.reduction_stride;
    coarse.accumulate = false;

    memops_extent_type stride = 1;

    for (size_t i = 0; i < description.num_dims; i++) {
        coarse.add_dimension(description.dims[i].extent, description.dims[i].input_stride, stride);

        stride = checked_mul<memops_extent_type>(stride, description.dims[i].extent);
    }

    coarse.add_dimension(num_chunks, chunk_stride, stride);

    ReductionDescription fine(description.dtype, description.operation);
    fine.input_offset = 0;
    fine.output_offset = description.output_offset;
    fine.reduction_extent = num_chunks;
    fine.reduction_stride = stride;
    fine.accumulate = description.accumulate;

    for (size_t i = 0; i < description.num_dims; i++) {
        fine.add_dimension(
            description.dims[i].extent,
            coarse.dims[i].output_stride,
            description.dims[i].output_stride
        );
    }

    ReductionDescription remainder(description.dtype, description.operation);
    remainder.input_offset = description.input_offset + num_chunks * description.reduction_stride;
    remainder.output_offset = description.output_offset;
    remainder.reduction_extent = remaining_chunks;
    remainder.reduction_stride = description.reduction_stride;
    remainder.accumulate = true;

    for (size_t i = 0; i < description.num_dims; i++) {
        remainder.add_dimension(
            description.dims[i].extent,
            description.dims[i].input_stride,
            description.dims[i].output_stride
        );
    }
}

size_t reduce_gpu_scratch_size(const ReductionDescription& description) {
    // The multi-level reduction path (the only consumer of scratch memory) is not enabled yet,
    // so `reduce_gpu` currently needs no scratch buffer for any description.
    (void)description;
    return 0;
}

void reduce_gpu(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    void* scratch_addr,
    const ReductionDescription& description
) {
    // Unused until the multi-level reduction path is enabled (see `reduce_gpu_scratch_size`).
    (void)scratch_addr;

    ReductionDescription simplified = description.simplify();

    if (simplified.is_noop()) {
        return;
    }

    if (simplified.is_equivalent_to_copy()) {
        copy_gpu(stream, src_addr, dst_addr, simplified.as_copy());
        return;
    }

    if (simplified.is_equivalent_to_fill()) {
        fill_gpu(stream, dst_addr, simplified.as_fill());
        return;
    }
    //
    //    if (simplified.reduction_extent > 1024 * 256) {
    //        do_multilevel_reduce(
    //            stream,
    //            src_addr,
    //            dst_addr,
    //            description
    //        );
    //    }

    do_reduce_gpu(stream, src_addr, dst_addr, simplified);
}

}  // namespace kmm::memops
