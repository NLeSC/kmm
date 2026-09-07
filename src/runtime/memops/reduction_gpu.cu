#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>

#include "memops_gpu_kernels.cuh"

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/fast_divisor.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/copy_gpu.hpp"
#include "kmm/runtime/memops/fill_gpu.hpp"
#include "kmm/runtime/memops/reducer.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/runtime/memops/reduction_gpu.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm::memops {

struct ReductionGPU: ReductionDescription {
    g_stream_t stream;
    const void* src_addr;
    void* dst_addr;

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
            if (!is_divisible(dims[i].input_stride, alignof(T))) {
                return false;
            }

            if (!is_divisible(dims[i].output_stride, alignof(T))) {
                return false;
            }
        }

        return true;
    }
};

ReductionGPU make_plan(
    g_stream_t stream,
    const void* src_base,
    void* dst_base,
    const ReductionDescription& description
) {
    ReductionGPU plan;
    plan.dtype = description.dtype;
    plan.operation = description.operation;
    plan.stream = stream;
    plan.src_addr = static_cast<const std::byte*>(src_base) + description.input_offset;
    plan.dst_addr = static_cast<std::byte*>(dst_base) + description.output_offset;
    plan.accumulate = description.accumulate;
    plan.reduction_stride = description.reduction_stride;
    plan.reduction_extent = description.reduction_extent > 0 ? description.reduction_extent : 0;
    plan.num_dims = 0;

    size_t old_rank = description.num_dims;
    size_t new_rank = 0;
    ReductionDim dims[MEMOPS_MAX_DIMS] = {};
    std::copy_n(description.dims, old_rank, dims);

    // Sorted by ascending `input_stride` (rather than `output_stride`): the axis at index 0 is
    // read repeatedly (once per reduction step, in every kernel that consults it for coalescing),
    // while its output element is written only once, so minimizing the input-side stride at index
    // 0 matters more than minimizing the output-side one.
    for (size_t i = 0; i < old_rank; i++) {
        for (size_t j = i + 1; j < old_rank; j++) {
            if (unsigned_abs(dims[j].input_stride) < unsigned_abs(dims[i].input_stride)) {
                std::swap(dims[i], dims[j]);
            }
        }

        auto new_dim = dims[i];

        // no reduction needed, just say there is one dim with extent==0
        if (new_dim.extent <= 0) {
            plan.dims[0].extent = 0;
            plan.dims[0].input_stride = 0;
            plan.dims[0].output_stride = 0;
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

            if (is_equal(new_dim.output_stride, plan.dims[k].output_stride * plan.dims[k].extent)
                && is_equal(
                    new_dim.input_stride,
                    plan.dims[k].input_stride * plan.dims[k].extent
                )) {
                plan.dims[new_rank - 1].extent *= new_dim.extent;
                continue;
            }

            if (is_divisible(new_dim.output_stride, plan.dims[k].output_stride)
                && is_divisible(new_dim.input_stride, plan.dims[k].input_stride)) {
                // TODO: should be something smart when the strides are multiples of each other
            }
        }

        plan.dims[new_rank] = new_dim;
        new_rank++;
    }

    plan.num_dims = new_rank;
    return plan;
}

// Picks which of the three reduction kernels to launch for one chunked reduction.
ReductionKernelKind select_reduction_kernel_kind(
    memops_extent_type reduction_extent,
    ptrdiff_t reduction_stride,
    ptrdiff_t input_stride0,
    ptrdiff_t output_stride0,
    size_t element_size
) {
    static constexpr int cooperative_threshold = 256;
    static constexpr size_t nearly_contiguous_factor = 4;

    if (reduction_extent < cooperative_threshold) {
        return ReductionKernelKind::Elementwise;
    }

    size_t nearly_contiguous_bound = nearly_contiguous_factor * element_size;
    bool reduction_axis_contiguous = unsigned_abs(reduction_stride) <= nearly_contiguous_bound;
    bool output_axis_contiguous = unsigned_abs(input_stride0) <= nearly_contiguous_bound
        && unsigned_abs(output_stride0) <= nearly_contiguous_bound;

    if (reduction_axis_contiguous && !output_axis_contiguous) {
        return ReductionKernelKind::Blockwise;
    }

    return ReductionKernelKind::Warpwise;
}

template<typename Reduction, size_t N>
void launch_reduction_kernel_rank_recur(
    g_stream_t stream,
    const std::byte* src_addr,
    std::byte* dst_addr,
    bool accumulate,
    memops_extent_type reduction_extent,
    ptrdiff_t reduction_stride,
    Vec<size_t, N> extents,
    Vec<ptrdiff_t, N> input_strides,
    Vec<ptrdiff_t, N> output_strides
) {
    uint32_t num_outputs = 1;
    Vec<uint32_t, N> region;

    for (size_t i = 0; i < N; i++) {
        if (extents[i] <= 0) {
            return;
        }

        uint32_t chunk = IndexMapper<uint32_t, N>::max_volume / num_outputs;

        while (extents[i] > chunk) {
            Vec<size_t, N> head = extents;
            head[i] = chunk;

            launch_reduction_kernel_rank_recur<Reduction, N>(
                stream,
                src_addr,
                dst_addr,
                accumulate,
                reduction_extent,
                reduction_stride,
                head,
                input_strides,
                output_strides
            );

            src_addr += chunk * input_strides[i];
            dst_addr += chunk * output_strides[i];
            extents[i] -= chunk;
        }

        region[i] = static_cast<uint32_t>(extents[i]);  // safe since 0 < extent[i] <= max_volume
        num_outputs *= region[i];
    }

    auto mapper = IndexMapper<uint32_t, N>(region);

    // A common case is that one buffer is 'folded' into another buffer. This means that for the parameters
    // N=1, accumulate=true, reduction_extent=1 we have a special case kernel.
    if constexpr (N == 1) {
        if (accumulate && reduction_extent == 1) {
            static constexpr uint32_t block_size = 256;
            static constexpr uint32_t max_blocks = 4096;
            uint32_t grid_size = std::min(max_blocks, div_ceil(num_outputs, block_size));

            elementwise_fold_kernel<Reduction, N, block_size><<<grid_size, block_size, 0, stream>>>(
                src_addr,
                dst_addr,
                mapper,
                input_strides,
                output_strides
            );
            return;
        }
    }

    size_t element_size = sizeof(typename Reduction::element_type);
    ReductionKernelKind kernel_kind = select_reduction_kernel_kind(
        reduction_extent,
        reduction_stride,
        N > 0 ? input_strides[0] : ptrdiff_t(element_size),
        N > 0 ? output_strides[0] : ptrdiff_t(element_size),
        element_size
    );

    static constexpr uint32_t threads_per_block = 256;
    static constexpr uint32_t max_blocks = 4096;
    uint32_t outputs_per_block;

    if (kernel_kind == ReductionKernelKind::Elementwise) {
        outputs_per_block = threads_per_block;
    } else if (kernel_kind == ReductionKernelKind::Warpwise) {
        outputs_per_block = KMM_REDUCE_WARP_SIZE;
    } else {
        outputs_per_block = 1;
    }

    uint32_t reduce_rows = threads_per_block / outputs_per_block;

    // `elementwise_reduce_kernel`'s cross-row combine folds pairs of rows by powers of two, so
    // `outputs_per_block` (== `blockDim.x`) must be a power of two dividing `threads_per_block`.
    KMM_ASSERT(is_power_of_two(outputs_per_block));
    KMM_ASSERT(outputs_per_block * reduce_rows == threads_per_block);

    dim3 block_size = {outputs_per_block, reduce_rows};
    dim3 grid_size = std::min(max_blocks, div_ceil(num_outputs, outputs_per_block));

    elementwise_reduce_kernel<Reduction, N, threads_per_block>
        <<<grid_size, block_size, 0, stream>>>(
            src_addr,
            dst_addr,
            accumulate,
            checked_cast<uint32_t>(reduction_extent),
            reduction_stride,
            mapper,
            input_strides,
            output_strides,
            kernel_kind
        );
}

template<typename Reduction, size_t N>
void launch_reduction_kernel_rank(const ReductionGPU& plan) {
    KMM_ASSERT(plan.num_dims == N);
    Vec<size_t, N> extents;
    Vec<ptrdiff_t, N> input_strides;
    Vec<ptrdiff_t, N> output_strides;

    for (size_t i = 0; i < N; i++) {
        extents[i] = checked_cast<size_t>(plan.dims[i].extent);
        input_strides[i] = plan.dims[i].input_stride;
        output_strides[i] = plan.dims[i].output_stride;
    }

    launch_reduction_kernel_rank_recur<Reduction>(
        plan.stream,
        reinterpret_cast<const std::byte*>(plan.src_addr),
        reinterpret_cast<std::byte*>(plan.dst_addr),
        plan.accumulate,
        plan.reduction_extent,
        plan.reduction_stride,
        extents,
        input_strides,
        output_strides
    );
}

template<ReductionOp Op, DataType dtype>
void launch_reduction_kernel_op(const ReductionGPU& plan) {
    using T = element_type_t<dtype>;
    size_t num_dims = plan.num_dims;

    if constexpr (!is_reduction_supported<T, Op>) {
        throw std::runtime_error(
            "invalid reduction parameters: unsupported operation for data type"
        );
    } else {
        if (!plan.is_aligned<T>()) {
            throw std::runtime_error(
                "invalid reduction parameters: address not aligned for data type"
            );
        }

        if (num_dims == 0) {
            ReductionGPU p = plan;
            p.num_dims = 1;
            p.dims[0] = ReductionDim {};
            launch_reduction_kernel_rank<Reducer<T, Op>, 1>(p);
        } else if (num_dims == 1) {
            launch_reduction_kernel_rank<Reducer<T, Op>, 1>(plan);
        } else if (num_dims == 2) {
            launch_reduction_kernel_rank<Reducer<T, Op>, 2>(plan);
        } else if (num_dims == 3) {
            launch_reduction_kernel_rank<Reducer<T, Op>, 3>(plan);
        } else {
            throw std::runtime_error("dimensionality of reduction is too high");
        }
    }
}

template<DataType dtype>
void launch_reduction_kernel_typed(const ReductionGPU& plan, ReductionOp op) {
    static constexpr DataType unsigned_dtype =  //
        dtype == DataType::Int32 ? DataType::Uint32
                                 : (dtype == DataType::Int64 ? DataType::Uint64 : dtype);

    switch (op) {
        case ReductionOp::Product:
            return launch_reduction_kernel_op<ReductionOp::Product, dtype>(plan);
        case ReductionOp::Min:
            return launch_reduction_kernel_op<ReductionOp::Min, dtype>(plan);
        case ReductionOp::Max:
            return launch_reduction_kernel_op<ReductionOp::Max, dtype>(plan);
        // for these operations, the operation is equivalent on unsigned and signed integers. We use the unsigned
        // dtype if possible to minimize the number of reduction kernels that are generated.
        case ReductionOp::Sum:
            return launch_reduction_kernel_op<ReductionOp::Sum, unsigned_dtype>(plan);
        case ReductionOp::BitwiseAnd:
            return launch_reduction_kernel_op<ReductionOp::BitwiseAnd, unsigned_dtype>(plan);
        case ReductionOp::BitwiseOr:
            return launch_reduction_kernel_op<ReductionOp::BitwiseOr, unsigned_dtype>(plan);
        default:
            throw std::runtime_error("invalid operation for reduction");
    }
}

void launch_reduction_kernel(const ReductionGPU& plan, DataType dtype, ReductionOp op) {
    switch (dtype) {
        case DataType::Int32:
            launch_reduction_kernel_typed<DataType::Int32>(plan, op);
            break;
        case DataType::Int64:
            launch_reduction_kernel_typed<DataType::Int64>(plan, op);
            break;
        case DataType::Uint32:
            launch_reduction_kernel_typed<DataType::Uint32>(plan, op);
            break;
        case DataType::Uint64:
            launch_reduction_kernel_typed<DataType::Uint64>(plan, op);
            break;
        case DataType::Float32:
            launch_reduction_kernel_typed<DataType::Float32>(plan, op);
            break;
        case DataType::Float64:
            launch_reduction_kernel_typed<DataType::Float64>(plan, op);
            break;
        case DataType::KeyValueInt64:
            launch_reduction_kernel_typed<DataType::KeyValueInt64>(plan, op);
            break;
        case DataType::KeyValueFloat64:
            launch_reduction_kernel_typed<DataType::KeyValueFloat64>(plan, op);
            break;
        default:
            throw std::runtime_error("invalid data type for reduction");
    }
}

constexpr uint32_t min_blocks_for_full_occupancy = 2048;
constexpr size_t reduction_scratch_budget = 16 * 1024 * 1024;
constexpr int32_t min_items_per_chunk = 256;

void launch_multilevel_reduction(
    const ReductionGPU& plan,
    int64_t num_chunks,
    size_t num_outputs,
    void* scratch_addr
) {
    KMM_ASSERT(scratch_addr != nullptr);

    size_t element_size = data_type_size(plan.dtype);
    size_t num_dims = plan.num_dims;

    // Round the chunk size *down* so that `items_per_chunk * num_chunks <= reduction_extent`
    auto items_per_chunk = plan.reduction_extent / num_chunks;
    auto tail_extent = plan.reduction_extent - items_per_chunk * num_chunks;

    // Dense (contiguous) strides used to lay out `num_chunks` partial results per output element in scratch.
    ptrdiff_t dense_strides[MEMOPS_MAX_DIMS] = {};
    ptrdiff_t dense_stride = ptrdiff_t(element_size);

    for (size_t i = 0; i < num_dims; i++) {
        dense_strides[i] = dense_stride;
        dense_stride *= plan.dims[i].extent;
    }

    ptrdiff_t chunk_output_stride = checked_mul<ptrdiff_t>(num_outputs, element_size);
    ptrdiff_t chunk_input_stride = checked_mul<ptrdiff_t>(items_per_chunk, plan.reduction_stride);

    // Kernel 1: reduce all `num_chunks` equally-sized chunks (each `items_per_chunk` elements) into
    // `scratch_addr`, adding the chunk index as an extra batch axis.
    {
        size_t chunk_pos = num_dims;

        // find the location to inject the chunk_input_stride
        for (size_t i = 0; i < num_dims; i++) {
            if (unsigned_abs(chunk_input_stride) < unsigned_abs(plan.dims[i].input_stride)) {
                chunk_pos = i;
                break;
            }
        }

        ReductionGPU chunk_plan;
        chunk_plan.stream = plan.stream;
        chunk_plan.src_addr = plan.src_addr;
        chunk_plan.dst_addr = scratch_addr;
        chunk_plan.accumulate = false;
        chunk_plan.reduction_extent = items_per_chunk;
        chunk_plan.reduction_stride = plan.reduction_stride;
        chunk_plan.num_dims = num_dims + 1;

        for (size_t i = 0; i < num_dims; i++) {
            size_t pos = i < chunk_pos ? i : i + 1;
            chunk_plan.dims[pos].extent = plan.dims[i].extent;
            chunk_plan.dims[pos].input_stride = plan.dims[i].input_stride;
            chunk_plan.dims[pos].output_stride = dense_strides[i];
        }

        chunk_plan.dims[chunk_pos].extent = num_chunks;
        chunk_plan.dims[chunk_pos].input_stride = chunk_input_stride;
        chunk_plan.dims[chunk_pos].output_stride = chunk_output_stride;

        launch_reduction_kernel(chunk_plan, plan.dtype, plan.operation);
    }

    // Kernel 2: reduce the leftover tail (`tail_extent < num_chunks` elements).
    if (tail_extent > 0) {
        ReductionGPU remainder_plan;
        remainder_plan.stream = plan.stream;
        remainder_plan.src_addr = static_cast<const std::byte*>(plan.src_addr)
            + checked_mul<ptrdiff_t>(items_per_chunk, num_chunks) * plan.reduction_stride;
        remainder_plan.dst_addr = plan.dst_addr;
        remainder_plan.accumulate = plan.accumulate;
        remainder_plan.reduction_extent = tail_extent;
        remainder_plan.reduction_stride = plan.reduction_stride;
        remainder_plan.num_dims = num_dims;

        for (size_t i = 0; i < num_dims; i++) {
            remainder_plan.dims[i] = plan.dims[i];
        }

        launch_reduction_kernel(remainder_plan, plan.dtype, plan.operation);
    }

    // Kernel 3: fold the `num_chunks` partial results in `scratch_addr` into the real output. When
    // Kernel 2 ran it already seeded `dst` (with the caller's `accumulate`), so accumulate on top
    // of it here; otherwise this pass is what honors the caller's `accumulate`.
    {
        ReductionGPU final_plan;
        final_plan.stream = plan.stream;
        final_plan.src_addr = scratch_addr;
        final_plan.dst_addr = plan.dst_addr;
        final_plan.accumulate = tail_extent > 0 ? true : plan.accumulate;
        final_plan.reduction_extent = num_chunks;
        final_plan.reduction_stride = chunk_output_stride;
        final_plan.num_dims = num_dims;

        for (size_t i = 0; i < num_dims; i++) {
            final_plan.dims[i].extent = plan.dims[i].extent;
            final_plan.dims[i].input_stride = dense_strides[i];
            final_plan.dims[i].output_stride = plan.dims[i].output_stride;
        }

        launch_reduction_kernel(final_plan, plan.dtype, plan.operation);
    }
}

// Conservative occupancy estimate: assumes the worst case of one block per output (as
// `blockwise_reduce_kernel` does), since that's the regime where extra chunks matter most -
// `elementwise_reduce_kernel`/`warpwise_reduce_kernel` already get plenty of blocks from the
// output dimension alone whenever `num_outputs` is large.
memops_extent_type plan_reduction_chunks(
    memops_extent_type reduction_extent,
    size_t num_outputs,
    size_t element_size
) {
    auto max_chunks_by_extent = reduction_extent / min_items_per_chunk;

    if (num_outputs >= min_blocks_for_full_occupancy || max_chunks_by_extent <= 1
        || num_outputs == 0) {
        return 1;
    }

    memops_extent_type wanted =
        div_ceil<memops_extent_type>(min_blocks_for_full_occupancy, num_outputs);
    memops_extent_type budget_limit =
        reduction_scratch_budget / std::max<size_t>(num_outputs * element_size, 1);

    return std::max<memops_extent_type>(1, std::min({wanted, budget_limit, max_chunks_by_extent}));
}

void reduce_gpu(
    g_stream_t stream,
    const void* src_base,
    void* dst_base,
    void* scratch_addr,
    const ReductionDescription& description
) {
    auto simplified = description;

    if (simplified.is_noop()) {
        return;
    }

    if (simplified.is_equivalent_to_copy()) {
        return copy_gpu(stream, src_base, dst_base, simplified.as_copy());
    }

    if (simplified.is_equivalent_to_fill()) {
        return fill_gpu(stream, dst_base, simplified.as_fill());
    }

    auto plan = make_plan(stream, src_base, dst_base, simplified);

    size_t element_size = data_type_size(simplified.dtype);
    size_t num_outputs = 1;

    for (size_t i = 0; i < plan.num_dims; i++) {
        num_outputs = checked_mul<size_t>(num_outputs, plan.dims[i].extent);
    }

    auto num_chunks = plan_reduction_chunks(plan.reduction_extent, num_outputs, element_size);

    if (num_chunks <= 1 || plan.num_dims == MEMOPS_MAX_DIMS) {
        launch_reduction_kernel(plan, plan.dtype, plan.operation);
    } else {
        launch_multilevel_reduction(plan, num_chunks, num_outputs, scratch_addr);
    }
}

size_t reduce_gpu_scratch_size(const ReductionDescription& description) {
    size_t element_size = data_type_size(description.dtype);
    auto num_outputs = description.num_outputs();
    auto reduction_extent = std::max<memops_extent_type>(description.reduction_extent, 0);

    auto num_chunks = plan_reduction_chunks(reduction_extent, num_outputs, element_size);

    if (num_chunks <= 1) {
        return 0;
    }

    return checked_mul<size_t>(num_chunks, num_outputs) * element_size;
}

}  // namespace kmm::memops
