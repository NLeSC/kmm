#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>

#include "memops_gpu_kernels.cuh"

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/fast_divisor.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/copy_gpu.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm::memops {

struct CopyPlan {
    const void* src_addr;
    void* dst_addr;
    size_t line_width;
    size_t num_dims;
    size_t extents[MEMOPS_MAX_DIMS + 1] = {};
    ptrdiff_t input_strides[MEMOPS_MAX_DIMS + 1] = {};
    ptrdiff_t output_strides[MEMOPS_MAX_DIMS + 1] = {};

    template<typename T>
    bool is_address_aligned() const {
        if (!is_divisible(reinterpret_cast<uintptr_t>(src_addr), alignof(T))) {
            return false;
        }

        if (!is_divisible(reinterpret_cast<uintptr_t>(dst_addr), alignof(T))) {
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

    template<typename T>
    bool is_aligned() const {
        if (!is_divisible(line_width, sizeof(T))) {
            return false;
        }

        return is_address_aligned<T>();
    }
};

CopyPlan make_plan(const void* src_base, void* dst_base, const CopyDescription& description) {
    CopyPlan plan;
    plan.num_dims = 0;
    plan.src_addr = static_cast<const std::byte*>(src_base) + description.src_offset;
    plan.dst_addr = static_cast<std::byte*>(dst_base) + description.dst_offset;
    plan.line_width = description.element_size;

    size_t old_rank = description.num_dims;
    size_t new_rank = 0;
    CopyDim dims[MEMOPS_MAX_DIMS] = {};
    std::copy_n(description.dims, old_rank, dims);

    for (size_t i = 0; i < old_rank; i++) {
        for (size_t j = i + 1; j < old_rank; j++) {
            if (unsigned_abs(dims[j].dst_stride) < unsigned_abs(dims[i].dst_stride)) {
                std::swap(dims[i], dims[j]);
            }
        }

        auto new_dim = dims[i];

        // no copy needed at all, just set line width to zero bytes.
        if (new_dim.extent <= 0) {
            plan.line_width = 0;
            new_rank = 0;
            break;
        }

        // skip this dimension if it has extent of one
        if (new_dim.extent == 1) {
            continue;
        }

        // if the dst_stride is zero, all values land at the same location. We can effectively consider its
        // extent to be equal to one.
        if (new_dim.dst_stride == 0) {
            // TODO: Maybe throw an exception here? Why would you want a dst stride of zero?
            continue;
        }

        // fix negative stride by subtracting offset from the pointer
        if (new_dim.dst_stride < 0) {
            plan.src_addr = static_cast<const std::byte*>(plan.src_addr)
                + (new_dim.extent - 1) * new_dim.src_stride;
            plan.dst_addr =
                static_cast<std::byte*>(plan.dst_addr) + (new_dim.extent - 1) * new_dim.dst_stride;

            new_dim.dst_stride = -new_dim.dst_stride;
            new_dim.src_stride = -new_dim.src_stride;
        }

        if (new_rank == 0) {
            // if the line width equals the stride, we can just extend the line width.
            if (is_equal(plan.line_width, new_dim.src_stride)
                && is_equal(plan.line_width, new_dim.dst_stride)) {
                plan.line_width *= checked_cast<size_t>(new_dim.extent);
                continue;
            }
        } else {
            auto k = new_rank - 1;

            if (is_equal(new_dim.dst_stride, plan.output_strides[k] * plan.extents[k])
                && is_equal(new_dim.src_stride, plan.input_strides[k] * plan.extents[k])) {
                plan.extents[new_rank - 1] *= new_dim.extent;
                continue;
            }

            if (is_divisible(new_dim.dst_stride, plan.output_strides[k])
                && is_divisible(new_dim.src_stride, plan.input_strides[k])) {
                // TODO: should be something smart when the strides are multiples of each other
            }
        }

        plan.extents[new_rank] = new_dim.extent;
        plan.input_strides[new_rank] = new_dim.src_stride;
        plan.output_strides[new_rank] = new_dim.dst_stride;
        new_rank++;
    }

    plan.num_dims = new_rank;
    return plan;
}

template<typename T, size_t N>
void launch_strided_kernel_rank_typed_recur(
    g_stream_t stream,
    Vec<size_t, N> extents,
    std::byte* dst_addr,
    const std::byte* src_addr,
    Vec<ptrdiff_t, N> dst_strides,
    Vec<ptrdiff_t, N> src_strides
) {
    constexpr uint32_t max_blocks = 1024;
    constexpr uint32_t threads_per_block = 256;

    // stride 0 must be contiguous
    KMM_ASSERT(dst_strides[0] == sizeof(T));
    KMM_ASSERT(src_strides[0] == sizeof(T));

    Vec<uint32_t, N> local_extents;
    uint32_t num_threads = 1;

    for (size_t i = 0; i < N; i++) {
        if (extents[i] == 0) {
            return;
        }

        // Largest count along axis `i` that keeps the running thread total within the range
        // that `FastDivisor` accepts as a numerator.
        uint32_t chunk = IndexMapper<uint32_t, N>::max_volume / num_threads;

        while (extents[i] > chunk) {
            Vec<size_t, N> head = extents;
            head[i] = chunk;

            launch_strided_kernel_rank_typed_recur<T>(
                stream,
                head,
                dst_addr,
                src_addr,
                dst_strides,
                src_strides
            );

            dst_addr += ptrdiff_t(chunk) * dst_strides[i];
            src_addr += ptrdiff_t(chunk) * src_strides[i];
            extents[i] -= chunk;
        }

        local_extents[i] = static_cast<uint32_t>(extents[i]);  // safe cast
        num_threads *= static_cast<uint32_t>(extents[i]);
    }

    uint32_t grid_size = std::min(div_ceil(num_threads, threads_per_block), max_blocks);
    auto mapper = IndexMapper<uint32_t, N>(local_extents);

    elementwise_copy_kernel<T, N><<<grid_size, threads_per_block, 0, stream>>>(
        dst_addr,
        src_addr,
        mapper,
        dst_strides,
        src_strides
    );
}

template<typename T, size_t N>
void launch_transpose_kernel_rank_typed_recur(
    g_stream_t stream,
    Vec<size_t, N> extents,
    std::byte* dst_addr,
    const std::byte* src_addr,
    Vec<ptrdiff_t, N> dst_strides,
    Vec<ptrdiff_t, N> src_strides
) {
    static_assert(N >= 2, "transpose kernel must have at least 2 dimensions");
    constexpr uint32_t max_blocks = 1024;
    constexpr uint32_t tile_size = 32;
    constexpr uint32_t block_dim_x = 32;
    constexpr uint32_t block_dim_y = 8;

    Vec<uint32_t, N> grid_extents;
    uint32_t num_blocks = 1;

    for (size_t i = 0; i < 2; i++) {
        if (extents[i] == 0) {
            return;
        }

        size_t count = div_ceil(extents[i], size_t(tile_size));
        uint32_t chunk = IndexMapper<uint32_t, N>::max_volume / num_blocks;

        chunk = std::min(chunk, IndexMapper<uint32_t, N>::max_volume / tile_size);

        while (count > chunk) {
            Vec<size_t, N> head = extents;
            head[i] = size_t(chunk) * tile_size;

            launch_transpose_kernel_rank_typed_recur<T>(
                stream,
                head,
                dst_addr,
                src_addr,
                dst_strides,
                src_strides
            );

            dst_addr += ptrdiff_t(head[i]) * dst_strides[i];
            src_addr += ptrdiff_t(head[i]) * src_strides[i];
            extents[i] -= head[i];
            count -= chunk;
        }

        grid_extents[i] = static_cast<uint32_t>(count);
        num_blocks *= static_cast<uint32_t>(count);
    }

    for (size_t i = 2; i < N; i++) {
        if (extents[i] == 0) {
            return;
        }

        size_t count = extents[i];
        uint32_t chunk = IndexMapper<uint32_t, N>::max_volume / num_blocks;

        while (count > chunk) {
            Vec<size_t, N> head = extents;
            head[i] = size_t(chunk);

            launch_transpose_kernel_rank_typed_recur<T>(
                stream,
                head,
                dst_addr,
                src_addr,
                dst_strides,
                src_strides
            );

            dst_addr += ptrdiff_t(head[i]) * dst_strides[i];
            src_addr += ptrdiff_t(head[i]) * src_strides[i];
            extents[i] -= head[i];
            count -= chunk;
        }

        grid_extents[i] = static_cast<uint32_t>(count);
        num_blocks *= static_cast<uint32_t>(count);
    }

    uint32_t grid_size = std::min(num_blocks, max_blocks);
    auto mapper = IndexMapper<uint32_t, N>(grid_extents);

    transpose_copy_kernel<T, N, tile_size, block_dim_x, block_dim_y>
        <<<grid_size, dim3(block_dim_x, block_dim_y), 0, stream>>>(
            dst_addr,
            src_addr,
            static_cast<uint32_t>(extents[0]),
            static_cast<uint32_t>(extents[1]),
            mapper,
            dst_strides,
            src_strides
        );
}

template<typename T, size_t N>
void launch_strided_kernel_rank_typed(g_stream_t stream, const CopyPlan& plan) {
    KMM_ASSERT(plan.num_dims == N);
    KMM_ASSERT(plan.line_width % sizeof(T) == 0);

    Vec<size_t, N + 1> extents;
    Vec<ptrdiff_t, N + 1> src_strides;
    Vec<ptrdiff_t, N + 1> dst_strides;

    extents[0] = plan.line_width / sizeof(T);
    src_strides[0] = sizeof(T);
    dst_strides[0] = sizeof(T);

    for (size_t i = 0; i < N; i++) {
        extents[i + 1] = plan.extents[i];
        src_strides[i + 1] = plan.input_strides[i];
        dst_strides[i + 1] = plan.output_strides[i];
    }

    auto* dst_addr = static_cast<std::byte*>(plan.dst_addr);
    const auto* src_addr = static_cast<const std::byte*>(plan.src_addr);

    if constexpr (N >= 2) {
        // if N >= 2, we can check if this copy is actually a transposition. To detect this, we scan over the
        // strides and attempt to find the "unit" axes for the source and destination. This is the axis that meets
        // the following criteria:
        //  - Must be sufficiently large
        //  - The stride cannot be zero
        //  - The stride is contiguous (or very close to contiguous).
        //
        // If the unit source axis is different from the unit destination axis, then we have a transposition and we
        // call the special transposition kernel to handle this.
        const size_t minimum_length = 32;
        const size_t near_contiguous = 4 * sizeof(T);

        size_t src_unit = 0;
        size_t dst_unit = 0;

        for (size_t i = 1; i < N + 1; i++) {
            if (extents[i] >= minimum_length && src_strides[i] != 0
                && unsigned_abs(src_strides[i]) <= near_contiguous
                && (src_unit == 0
                    || unsigned_abs(src_strides[i]) < unsigned_abs(src_strides[src_unit]))) {
                src_unit = i;
            }
            if (extents[i] >= minimum_length && dst_strides[i] != 0
                && unsigned_abs(dst_strides[i]) <= near_contiguous
                && (dst_unit == 0
                    || unsigned_abs(dst_strides[i]) < unsigned_abs(dst_strides[dst_unit]))) {
                dst_unit = i;
            }
        }

        if (src_unit != 0 && dst_unit != 0 && src_unit != dst_unit) {
            // A genuine transpose never merges anything into the line-width axis, so it stays a
            // single element. In that case drop it and dispatch over the N real axes, so e.g. a
            // 2D transpose runs the rank-2 kernel rather than a rank-3 one with a trailing 1.
            const bool drop_line_axis = extents[0] == 1;

            // rotate the src-contiguous axis to position 0
            for (size_t i = src_unit; i > 0; i--) {
                std::swap(src_strides[i], src_strides[i - 1]);
                std::swap(dst_strides[i], dst_strides[i - 1]);
                std::swap(extents[i], extents[i - 1]);
            }

            // that rotation shifted every axis below src_unit up by one
            if (dst_unit < src_unit) {
                dst_unit += 1;
            }

            // rotate the dst-contiguous axis to position 1
            for (size_t i = dst_unit; i > 1; i--) {
                std::swap(src_strides[i], src_strides[i - 1]);
                std::swap(dst_strides[i], dst_strides[i - 1]);
                std::swap(extents[i], extents[i - 1]);
            }

            launch_transpose_kernel_rank_typed_recur<T>(
                stream,
                extents,
                dst_addr,
                src_addr,
                dst_strides,
                src_strides
            );

            return;
        }
    }

    launch_strided_kernel_rank_typed_recur<T>(
        stream,
        extents,
        dst_addr,
        src_addr,
        dst_strides,
        src_strides
    );
}

template<typename T>
void launch_strided_kernel_typed(g_stream_t stream, const CopyPlan& plan) {
    size_t num_dims = plan.num_dims;

    if (num_dims == 1) {
        launch_strided_kernel_rank_typed<T, 1>(stream, plan);
    } else if (num_dims == 2) {
        launch_strided_kernel_rank_typed<T, 2>(stream, plan);
    } else if (num_dims == 3) {
        launch_strided_kernel_rank_typed<T, 3>(stream, plan);
    } else if (num_dims == 4) {
        launch_strided_kernel_rank_typed<T, 4>(stream, plan);
    } else {
        // should not happen
        KMM_PANIC("invalid dimensionality");
    }
}

bool launch_strided_kernel(g_stream_t stream, const CopyPlan& plan) {
    if (plan.is_aligned<ulonglong2>()) {
        launch_strided_kernel_typed<ulonglong2>(stream, plan);
    } else if (plan.is_aligned<ulong>()) {
        launch_strided_kernel_typed<ulong>(stream, plan);
    } else if (plan.is_aligned<uint>()) {
        launch_strided_kernel_typed<uint>(stream, plan);
    } else if (plan.is_aligned<ushort>()) {
        launch_strided_kernel_typed<ushort>(stream, plan);
    } else {
        KMM_ASSERT(plan.is_aligned<std::byte>());
        launch_strided_kernel_typed<std::byte>(stream, plan);
    }

    return true;
}

void execute_copy_plan(g_stream_t stream, const CopyPlan& plan) {
    // nothing to do
    if (plan.line_width == 0) {
        return;
    }

    // simple 1D copy
    if (plan.num_dims == 0) {
        KMM_GPU_CHECK(g_memcpy_async(
            reinterpret_cast<g_device_ptr_t>(plan.dst_addr),
            reinterpret_cast<g_device_ptr_t>(const_cast<void*>(plan.src_addr)),
            plan.line_width,
            stream
        ));

        return;
    }

    // 2D copy (if possible)
    if (plan.num_dims == 1 && is_greater(plan.input_strides[0], plan.line_width)
        && is_greater(plan.output_strides[0], plan.line_width)) {
        gpu_memcpy2d_t p;
        ::memset(&p, 0, sizeof(gpu_memcpy2d_t));

        p.srcMemoryType = G_MEMORYTYPE_DEVICE;
        p.srcDevice = reinterpret_cast<g_device_ptr_t>(const_cast<void*>(plan.src_addr));
        p.srcPitch = checked_cast<size_t>(plan.input_strides[0]);
        p.dstMemoryType = G_MEMORYTYPE_DEVICE;
        p.dstDevice = reinterpret_cast<g_device_ptr_t>(plan.dst_addr);
        p.dstPitch = checked_cast<size_t>(plan.output_strides[0]);
        p.WidthInBytes = checked_cast<size_t>(plan.line_width);
        p.Height = checked_cast<size_t>(plan.extents[0]);

        KMM_GPU_CHECK(g_memcpy_2d_async(&p, stream));
        return;
    }

    launch_strided_kernel(stream, plan);
}

void copy_gpu(
    g_stream_t stream,
    const void* src_base,
    void* dst_base,
    const CopyDescription& description
) {
    auto plan = make_plan(src_base, dst_base, description);
    execute_copy_plan(stream, plan);
}

}  // namespace kmm::memops
