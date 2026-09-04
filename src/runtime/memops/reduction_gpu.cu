#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>

#include "reduction_traits.hpp"

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/runtime/memops/reduction_gpu.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm::memops {

struct ReductionPlan {
    const void* src_addr;
    void* dst_addr;
    bool accumulate;
    size_t reduction_extent;
    ptrdiff_t reduction_stride;
    size_t num_dims;
    size_t extents[MEMOPS_MAX_DIMS] = {};
    ptrdiff_t input_strides[MEMOPS_MAX_DIMS] = {};
    ptrdiff_t output_strides[MEMOPS_MAX_DIMS] = {};

    template<typename T>
    bool is_aligned() const {
        if (!is_divisible(reinterpret_cast<uintptr_t>(src_addr), alignof(T))) {
            return false;
        }

        if (!is_divisible(reinterpret_cast<uintptr_t>(dst_addr), alignof(T))) {
            return false;
        }

        if (!is_divisible(reduction_stride, alignof(T))) {
            return false;
        }

        for (size_t i = 0; i < num_dims; i++) {
            if (!is_divisible(input_strides[i], alignof(T))) {
                return false;
            }

            if (!is_divisible(output_strides[i], alignof(T))) {
                return false;
            }
        }

        return true;
    }
};

ReductionPlan make_plan(
    const void* src_base,
    void* dst_base,
    const ReductionDescription& description
) {
    ReductionPlan plan;
    plan.num_dims = 0;
    plan.src_addr = static_cast<const std::byte*>(src_base) + description.input_offset;
    plan.dst_addr = static_cast<std::byte*>(dst_base) + description.output_offset;
    plan.accumulate = description.accumulate;
    plan.reduction_stride = description.reduction_stride;
    plan.reduction_extent =
        description.reduction_extent > 0 ? size_t(description.reduction_extent) : 0;

    size_t old_rank = description.num_dims;
    size_t new_rank = 0;
    ReductionDim dims[MEMOPS_MAX_DIMS] = {};
    std::copy_n(description.dims, old_rank, dims);

    for (size_t i = 0; i < old_rank; i++) {
        for (size_t j = i + 1; j < old_rank; j++) {
            if (unsigned_abs(dims[j].output_stride) < unsigned_abs(dims[i].output_stride)) {
                std::swap(dims[i], dims[j]);
            }
        }

        auto new_dim = dims[i];

        // no reduction needed, just say there is one dim with extent==0
        if (new_dim.extent <= 0) {
            plan.extents[0] = 0;
            new_rank = 1;
            break;
        }

        // skip this dimension if it has extent of one
        if (new_dim.extent == 1) {
            continue;
        }

        // if the dst_stride is zero, all values land at the same location. We can effectively consider its
        // extent to be equal to one.
        if (new_dim.output_stride == 0) {
            // TODO: Maybe throw an exception here? Why would you want a dst stride of zero?
            continue;
        }

        // fix negative stride by subtracting offset from the pointer
        if (new_dim.output_stride < 0) {
            plan.src_addr = static_cast<const std::byte*>(plan.src_addr)
                + (new_dim.extent - 1) * new_dim.input_stride;
            plan.dst_addr = static_cast<std::byte*>(plan.dst_addr)
                + (new_dim.extent - 1) * new_dim.output_stride;

            new_dim.output_stride = -new_dim.output_stride;
            new_dim.input_stride = -new_dim.input_stride;
        }

        if (new_rank > 0) {
            auto k = new_rank - 1;

            if (is_equal(new_dim.output_stride, plan.output_strides[k] * plan.extents[k])
                && is_equal(new_dim.input_stride, plan.input_strides[k] * plan.extents[k])) {
                plan.extents[new_rank - 1] *= new_dim.extent;
                continue;
            }

            if (is_divisible(new_dim.output_stride, plan.output_strides[k])
                && is_divisible(new_dim.input_stride, plan.input_strides[k])) {
                // TODO: should be something smart when the strides are multiples of each other
            }
        }

        plan.extents[new_rank] = new_dim.extent;
        plan.input_strides[new_rank] = new_dim.input_stride;
        plan.output_strides[new_rank] = new_dim.output_stride;
        new_rank++;
    }

    plan.num_dims = new_rank;
    return plan;
}

template<uint BlockSize, typename T, ReductionOp Op, size_t Rank>
__global__ void blockwise_reduce_kernel(
    void* dst_addr,
    const void* src_addr,
    Vec<ptrdiff_t, Rank> input_strides,
    Vec<ptrdiff_t, Rank> output_strides,
    Vec<uint, Rank> extents,
    ptrdiff_t reduction_stride,
    uint reduction_extent,
    bool accumulate
) {
    uint p[3] = {blockIdx.x, blockIdx.y, blockIdx.z};

#pragma unroll
    for (size_t i = 0; i < Rank; i++) {
        dst_addr = static_cast<std::byte*>(dst_addr) + output_strides[i] * ptrdiff_t(p[i]);
        src_addr = static_cast<const std::byte*>(src_addr) + input_strides[i] * ptrdiff_t(p[i]);
    }

    T value = ReductionTraits<T, Op>::identity();

    for (uint i = threadIdx.x; i < reduction_extent; i++) {
        value = ReductionTraits<T, Op>::combine(*static_cast<const T*>(src_addr), value);
        src_addr = static_cast<const std::byte*>(src_addr) + reduction_stride;
    }

    KMM_PANIC("reduce block");

    if (threadIdx.x == 0) {
        if (accumulate) {
            value = ReductionTraits<T, Op>::combine(*static_cast<const T*>(dst_addr), value);
        }

        *static_cast<T*>(dst_addr) = value;
    }
}

template<typename T, ReductionOp Op>
void __global__ launch_reduction_kernel_blockwise(g_stream_t stream, const ReductionPlan& plan) {}

template<typename T, ReductionOp Op, size_t Rank>
__global__ void elementwise_reduce_kernel(
    void* dst_addr,
    const void* src_addr,
    Vec<ptrdiff_t, Rank> input_strides,
    Vec<ptrdiff_t, Rank> output_strides,
    Vec<uint, Rank> extents,
    ptrdiff_t reduction_stride,
    uint reduction_extent,
    bool accumulate
) {
    uint p[3] = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z,
    };

#pragma unroll
    for (size_t i = 0; i < Rank; i++) {
        if (p[i] >= extents[i]) {
            return;
        }

        dst_addr = static_cast<std::byte*>(dst_addr) + output_strides[i] * ptrdiff_t(p[i]);
        src_addr = static_cast<const std::byte*>(src_addr) + input_strides[i] * ptrdiff_t(p[i]);
    }

    T value = ReductionTraits<T, Op>::identity();

    for (uint i = 0; i < reduction_extent; i++) {
        value = ReductionTraits<T, Op>::combine(*static_cast<const T*>(src_addr), value);
        src_addr = static_cast<const std::byte*>(src_addr) + reduction_stride;
    }

    if (accumulate) {
        value = ReductionTraits<T, Op>::combine(*static_cast<const T*>(dst_addr), value);
    }

    *static_cast<T*>(dst_addr) = value;
}

// Choose a thread-block shape for `elementwise_copy_kernel`.
template<size_t Rank>
dim3 compute_block_size(
    size_t elem_size,
    Vec<ptrdiff_t, Rank> input_strides,
    Vec<ptrdiff_t, Rank> output_strides,
    Vec<size_t, Rank> extents
) {
    KMM_TODO();
}

template<typename T, ReductionOp Op>
void launch_reduction_kernel_elementwise(g_stream_t stream, const ReductionPlan& plan) {
    KMM_ASSERT(plan.is_aligned<T>());
    KMM_ASSERT(plan.num_dims <= 3);

    Vec<ptrdiff_t, 3> input_strides;
    Vec<ptrdiff_t, 3> output_strides;
    Vec<size_t, 3> extents;

    for (size_t i = 0; i < 3; i++) {
        input_strides[i] = i < plan.num_dims ? plan.input_strides[i] : 0;
        output_strides[i] = i < plan.num_dims ? plan.output_strides[i] : 0;
        extents[i] = i < plan.num_dims ? plan.extents[i] : 1u;
    }

    constexpr size_t TILE_X = size_t(1) << 31;
    constexpr size_t TILE_YZ = 65535;

    dim3 block_size = compute_block_size(sizeof(T), input_strides, output_strides, extents);

    size_t nx = extents[0];
    size_t ny = extents[1];
    size_t nz = extents[2];

    for (size_t ox = 0; ox < nx; ox += TILE_X) {
        for (size_t oy = 0; oy < ny; oy += TILE_YZ) {
            for (size_t oz = 0; oz < nz; oz += TILE_YZ) {
                size_t ex = std::min(TILE_X, nx - ox);
                size_t ey = std::min(TILE_YZ, ny - oy);
                size_t ez = std::min(TILE_YZ, nz - oz);

                auto* src_addr = static_cast<const std::byte*>(plan.src_addr);
                auto* dst_addr = static_cast<std::byte*>(plan.dst_addr);
                Vec<uint, 3> sub;

                if constexpr (true) {
                    src_addr += ptrdiff_t(ox) * input_strides[0];
                    dst_addr += ptrdiff_t(ox) * output_strides[0];
                    sub[0] = checked_cast<uint>(ex);
                }

                if constexpr (true) {
                    src_addr += ptrdiff_t(oy) * input_strides[1];
                    dst_addr += ptrdiff_t(oy) * output_strides[1];
                    sub[1] = checked_cast<uint>(ey);
                }

                if constexpr (true) {
                    src_addr += ptrdiff_t(oz) * input_strides[2];
                    dst_addr += ptrdiff_t(oz) * output_strides[2];
                    sub[2] = checked_cast<uint>(ez);
                }

                dim3 grid_size = {
                    checked_cast<uint>(div_ceil<size_t>(ex, block_size.x)),
                    checked_cast<uint>(div_ceil<size_t>(ey, block_size.y)),
                    checked_cast<uint>(div_ceil<size_t>(ez, block_size.z)),
                };

                elementwise_reduce_kernel<T, Op><<<grid_size, block_size, 0, stream>>>(
                    dst_addr,
                    src_addr,
                    input_strides,
                    output_strides,
                    sub,
                    plan.reduction_stride,
                    checked_cast<uint>(plan.reduction_extent),
                    plan.accumulate
                );
            }
        }
    }
}

template<typename T, ReductionOp Op>
void launch_reduction_kernel_op(g_stream_t stream, const ReductionPlan& plan) {
    size_t num_dims = plan.num_dims;

    if constexpr (!is_reduction_supported<T, Op>) {
        throw std::runtime_error(
            "invalid reduction parameters: unsupported operation for data type"
        );
    } else if (!plan.is_aligned<T>()) {
        throw std::runtime_error("invalid reduction parameters: address not aligned for data type");
    } else if (num_dims > 3) {
        // The strided kernel handles at most three axes (Rank == 3).
        size_t k = std::min_element(plan.extents, plan.extents + num_dims) - plan.extents;

        ReductionPlan sub = plan;
        sub.num_dims = num_dims - 1;

        // Drop axis `k` from the descriptor arrays, shifting the trailing axes down.
        for (size_t d = k; d + 1 < num_dims; d++) {
            sub.extents[d] = plan.extents[d + 1];
            sub.input_strides[d] = plan.input_strides[d + 1];
            sub.output_strides[d] = plan.output_strides[d + 1];
        }

        for (size_t i = 0; i < plan.extents[k]; i++) {
            sub.src_addr =
                static_cast<const std::byte*>(plan.src_addr) + ptrdiff_t(i) * plan.input_strides[k];
            sub.dst_addr =
                static_cast<std::byte*>(plan.dst_addr) + ptrdiff_t(i) * plan.output_strides[k];
            launch_reduction_kernel_op<T, Op>(stream, sub);

            // The next call MUST accumulate into the output of the previous call.
            sub.accumulate = true;
        }
    } else {
        KMM_TODO();

        //        if (plan.reduction_extent > 1024) {
        //            launch_reduction_kernel_blockwise<T, Op>(stream, plan);
        //        } else {
        //            launch_reduction_kernel_elementwise<T, Op>(stream, plan);
        //        }
    }
}

template<typename T>
void launch_reduction_kernel_typed(g_stream_t stream, const ReductionPlan& plan, ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
            launch_reduction_kernel_op<T, ReductionOp::Sum>(stream, plan);
            break;
        case ReductionOp::Product:
            launch_reduction_kernel_op<T, ReductionOp::Product>(stream, plan);
            break;
        case ReductionOp::Min:
            launch_reduction_kernel_op<T, ReductionOp::Min>(stream, plan);
            break;
        case ReductionOp::Max:
            launch_reduction_kernel_op<T, ReductionOp::Max>(stream, plan);
            break;
        case ReductionOp::BitwiseAnd:
            launch_reduction_kernel_op<T, ReductionOp::BitwiseAnd>(stream, plan);
            break;
        case ReductionOp::BitwiseOr:
            launch_reduction_kernel_op<T, ReductionOp::BitwiseOr>(stream, plan);
            break;
        default:
            KMM_PANIC("invalid operation for reduction");
    }
}

void launch_reduction_kernel(
    g_stream_t stream,
    const ReductionPlan& plan,
    DataType dtype,
    ReductionOp op
) {
    switch (dtype) {
        case DataType::Int32:
            launch_reduction_kernel_typed<int32_t>(stream, plan, op);
            break;
        case DataType::Int64:
            launch_reduction_kernel_typed<int64_t>(stream, plan, op);
            break;
        case DataType::Uint32:
            launch_reduction_kernel_typed<uint32_t>(stream, plan, op);
            break;
        case DataType::Uint64:
            launch_reduction_kernel_typed<uint64_t>(stream, plan, op);
            break;
        case DataType::Float32:
            launch_reduction_kernel_typed<float>(stream, plan, op);
            break;
        case DataType::Float64:
            launch_reduction_kernel_typed<double>(stream, plan, op);
            break;
        case DataType::KeyValueInt64:
            launch_reduction_kernel_typed<KeyValue<int64_t>>(stream, plan, op);
            break;
        case DataType::KeyValueFloat64:
            launch_reduction_kernel_typed<KeyValue<double>>(stream, plan, op);
            break;
        default:
            KMM_PANIC("invalid data type for reduction");
    }
}

void reduce_gpu(
    g_stream_t stream,
    const void* src_base,
    void* dst_base,
    void* scratch_addr,
    const ReductionDescription& description
) {
    auto plan = make_plan(src_base, dst_base, description);
    launch_reduction_kernel(stream, plan, description.dtype, description.operation);
}

size_t reduce_gpu_scratch_size(const ReductionDescription& description) {
    return 0;
}

}  // namespace kmm::memops
