#include <cstddef>
#include <cstdint>

#include "cub/block/block_reduce.cuh"

#include "kmm/core/const_value.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/reduction.hpp"

namespace kmm {

template <typename T, ReductionOp Op>
struct ReductionTraits;

template <typename T>
struct ReductionTraits<T, ReductionOp::Sum> {
    KMM_HOST_DEVICE static T identity() {
        return static_cast<T>(0);
    }

    KMM_HOST_DEVICE static T combine(T x, T y) {
        return x + y;
    }
};

template <typename T>
struct ReductionTraits<T, ReductionOp::Product> {
    KMM_HOST_DEVICE static T identity() {
        return static_cast<T>(1);
    }

    KMM_HOST_DEVICE static T combine(T x, T y) {
        return x * y;
    }
};

template <typename T>
struct ReductionTraits<T, ReductionOp::Min> {
    KMM_HOST_DEVICE static T identity() {
        return std::numeric_limits<T>::max();
    }

    KMM_HOST_DEVICE static T combine(T x, T y) {
        return x < y ? x : y;
    }
};

template <typename T>
struct ReductionTraits<T, ReductionOp::Max> {
    KMM_HOST_DEVICE static T identity() {
        return std::numeric_limits<T>::lowest();
    }

    KMM_HOST_DEVICE static T combine(T x, T y) {
        return x > y ? x : y;
    }
};

template <typename T, ReductionOp Op, size_t Rank>
void __global__ elementwise_reduce_kernel(
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

template <uint BlockDim, typename T, ReductionOp Op, size_t Rank>
void __global__ blockwise_reduce_kernel(
    const void* src_addr,
    void* dst_addr,
    Vec<ReductionDim, Rank> dims,
    memops_extent_type reduction_extent,
    memops_stride_type reduction_stride,
    bool accumulate
) {
    __shared__ typename cub::BlockReduce<T, BlockDim>::TempStorage smem_storage;
    cub::BlockReduce<T, BlockDim> block_reduce{smem_storage};

    uint p[3] = {
        blockIdx.x,
        blockIdx.y,
        blockIdx.z
    };

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

    blockwise_reduce_kernel<block_size, T, Op><<<
        grid_size,
        block_size,
        0,
        stream
    >>>(
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
        return launch_elementwise_reduce<T, Op>(
            stream,
            src_addr,
            dst_addr,
            description
        );
    } else {
        return launch_blockwise_reduce<T, Op>(
            stream,
            src_addr,
            dst_addr,
            description
        );
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
            return reduced_typed_op_async<T, ReductionOp::Sum>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::Product:
            return reduced_typed_op_async<T, ReductionOp::Product>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::Min:
            return reduced_typed_op_async<T, ReductionOp::Min>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case ReductionOp::Max:
            return reduced_typed_op_async<T, ReductionOp::Max>(
                stream,
                src_addr,
                dst_addr,
                description
            );
    }

    KMM_PANIC("invalid reduction operator");
}

void do_reduce_async(
    g_stream_t stream,
        const void* src_addr,
        void* dst_addr,
        const ReductionDescription& description
) {
    switch (description.dtype) {
        case DataType::Unknown:
            break;
        case DataType::Int8:
            reduce_typed_async<int8_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Int16:
            reduce_typed_async<int16_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Int32:
            reduce_typed_async<int32_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Int64:
            reduce_typed_async<int64_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Uint8:
            reduce_typed_async<uint8_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Uint16:
            reduce_typed_async<uint16_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Uint32:
            reduce_typed_async<uint32_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Uint64:
            reduce_typed_async<uint64_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Float32:
            reduce_typed_async<float>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
        case DataType::Float64:
            reduce_typed_async<double>(
                stream,
                src_addr,
                dst_addr,
                description
            );
            break;
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
        coarse.add_dimension(
            description.dims[i].extent,
            description.dims[i].input_stride,
            stride
        );

        stride = checked_mul<memops_extent_type>(stride, description.dims[i].extent);
    }

    coarse.add_dimension(
        num_chunks,
        chunk_stride,
        stride
    );

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
    remainder.input_offset = description.input_offset + remaining_chunks * description.reduction_stride;
    remainder.output_offset = description.output_offset;
    remainder.reduction_extent = description.reduction_extent % elements_per_block;
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

void reduce_async(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    ReductionDescription simplified = description.simplify();

    if (simplified.is_equivalent_to_copy()) {
        copy_async(stream, src_addr, dst_addr, simplified.as_copy());
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

    do_reduce_async(
        stream, src_addr, dst_addr, simplified
    );
}

}  // namespace kmm
