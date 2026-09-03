#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>

#include "kmm/core/checked_compare.hpp"
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
    size_t extents[MEMOPS_MAX_DIMS] = {};
    ptrdiff_t input_strides[MEMOPS_MAX_DIMS] = {};
    ptrdiff_t output_strides[MEMOPS_MAX_DIMS] = {};

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

template<typename T, size_t Rank>
__global__ void elementwise_copy_kernel(
    void* dst_addr,
    const void* src_addr,
    Vec<ptrdiff_t, Rank> input_strides,
    Vec<ptrdiff_t, Rank> output_strides,
    Vec<uint, Rank> extents
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

    *static_cast<T*>(dst_addr) = *static_cast<const T*>(src_addr);
}

// Choose a thread-block shape for `elementwise_copy_kernel`.
template<size_t Rank>
dim3 compute_block_size(
    size_t elem_size,
    Vec<ptrdiff_t, Rank> input_strides,
    Vec<ptrdiff_t, Rank> output_strides,
    Vec<size_t, Rank> extents
) {
    constexpr size_t BYTES_PER_LINE = 128;
    constexpr size_t THREADS_PER_BLOCK = 512;

    // Distinct cache lines touched by `n` elements spaced `stride` bytes apart.
    auto lines = [&](size_t n, size_t stride) -> size_t {
        if (n <= 1) {
            return 1;
        }

        size_t span = (n - 1) * stride + elem_size;
        return std::min(n, div_ceil<size_t>(span, BYTES_PER_LINE));
    };

    // upper bound on the block extent per axis.
    size_t cap[3] = {1, 1, 1};
    for (size_t i = 0; i < Rank; i++) {
        cap[i] = round_up_to_power_of_two(std::min(extents[i], THREADS_PER_BLOCK));
    }

    // Z-axis cannot exceed 64.
    if constexpr (Rank >= 3) {
        cap[2] = std::min(cap[2], size_t(64));
    }

    dim3 best = {1, 1, 1};
    size_t best_ws = 1;  // best `ws / vol` seen so far, kept as a fraction
    size_t best_vol = 0;  // 0 marks "unset", i.e. compares as +infinity

    for (size_t bz = 1; bz <= cap[2]; bz *= 2) {
        for (size_t by = 1; by <= cap[1]; by *= 2) {
            for (size_t bx = 1; bx <= cap[0]; bx *= 2) {
                size_t vol = bx * by * bz;
                if (vol > THREADS_PER_BLOCK) {
                    break;
                }

                size_t b[3] = {bx, by, bz};
                size_t ws_in = 1;
                size_t ws_out = 1;

                for (size_t i = 0; i < Rank; i++) {
                    ws_in *= lines(b[i], unsigned_abs(input_strides[i]));
                    ws_out *= lines(b[i], unsigned_abs(output_strides[i]));
                }

                size_t ws = ws_in + ws_out;

                // Compare `ws / vol` by cross-multiplication; break ties toward the more
                // occupied block, then toward more threads on `x`.
                bool better = ws * best_vol < best_ws * vol;
                bool tie = ws * best_vol == best_ws * vol;

                if (better || (tie && vol > best_vol) || (tie && vol == best_vol && bx > best.x)) {
                    best = {checked_cast<uint>(bx), checked_cast<uint>(by), checked_cast<uint>(bz)};
                    best_ws = ws;
                    best_vol = vol;
                }
            }
        }
    }

    return best;
}

template<typename T, size_t Rank>
void launch_strided_kernel_rank_typed(g_stream_t stream, const CopyPlan& plan) {
    static_assert(Rank >= 1 && Rank <= 3, "invalid rank");
    KMM_ASSERT(plan.line_width == sizeof(T));

    Vec<ptrdiff_t, Rank> input_strides;
    Vec<ptrdiff_t, Rank> output_strides;
    Vec<size_t, Rank> extents;

    for (size_t i = 0; i < Rank; i++) {
        input_strides[i] = plan.input_strides[i];
        output_strides[i] = plan.output_strides[i];
        extents[i] = plan.extents[i];
    }

    constexpr size_t TILE_X = size_t(1) << 31;
    constexpr size_t TILE_YZ = 65535;

    dim3 block_size = compute_block_size<Rank>(sizeof(T), input_strides, output_strides, extents);

    size_t nx = extents[0];
    size_t ny = Rank >= 2 ? plan.extents[1] : 1;
    size_t nz = Rank >= 3 ? plan.extents[2] : 1;

    for (size_t ox = 0; ox < nx; ox += TILE_X) {
        for (size_t oy = 0; oy < ny; oy += TILE_YZ) {
            for (size_t oz = 0; oz < nz; oz += TILE_YZ) {
                size_t ex = std::min(TILE_X, nx - ox);
                size_t ey = std::min(TILE_YZ, ny - oy);
                size_t ez = std::min(TILE_YZ, nz - oz);

                auto* src_addr = static_cast<const std::byte*>(plan.src_addr);
                auto* dst_addr = static_cast<std::byte*>(plan.dst_addr);
                Vec<uint, Rank> sub;

                if constexpr (Rank >= 1) {
                    src_addr += ptrdiff_t(ox) * input_strides[0];
                    dst_addr += ptrdiff_t(ox) * output_strides[0];
                    sub[0] = checked_cast<uint>(ex);
                }

                if constexpr (Rank >= 2) {
                    src_addr += ptrdiff_t(oy) * input_strides[1];
                    dst_addr += ptrdiff_t(oy) * output_strides[1];
                    sub[1] = checked_cast<uint>(ey);
                }

                if constexpr (Rank >= 3) {
                    src_addr += ptrdiff_t(oz) * input_strides[2];
                    dst_addr += ptrdiff_t(oz) * output_strides[2];
                    sub[2] = checked_cast<uint>(ez);
                }

                dim3 grid_size = {
                    checked_cast<uint>(div_ceil<size_t>(ex, block_size.x)),
                    checked_cast<uint>(div_ceil<size_t>(ey, block_size.y)),
                    checked_cast<uint>(div_ceil<size_t>(ez, block_size.z)),
                };

                elementwise_copy_kernel<T, Rank><<<grid_size, block_size, 0, stream>>>(
                    dst_addr,
                    src_addr,
                    input_strides,
                    output_strides,
                    sub
                );
            }
        }
    }
}

template<typename T>
void launch_strided_kernel_typed(g_stream_t stream, const CopyPlan& plan) {
    size_t num_dims = plan.num_dims;

    // if line_width != sizeof(T), inject a new axis at the beginning.
    if (plan.line_width != sizeof(T) && num_dims < MEMOPS_MAX_DIMS) {
        CopyPlan p = plan;
        p.line_width = sizeof(T);
        p.num_dims = num_dims + 1;

        // insert at the start
        p.extents[0] = plan.line_width / sizeof(T);
        p.input_strides[0] = sizeof(T);
        p.output_strides[0] = sizeof(T);

        // shift all one forward
        std::copy_n(plan.extents, num_dims, p.extents + 1);
        std::copy_n(plan.input_strides, num_dims, p.input_strides + 1);
        std::copy_n(plan.output_strides, num_dims, p.output_strides + 1);

        return launch_strided_kernel_typed<T>(stream, p);
    }

    KMM_ASSERT(num_dims >= 1);

    if (num_dims == 1) {
        launch_strided_kernel_rank_typed<T, 1>(stream, plan);
    } else if (num_dims == 2) {
        launch_strided_kernel_rank_typed<T, 2>(stream, plan);
    } else if (num_dims == 3) {
        launch_strided_kernel_rank_typed<T, 3>(stream, plan);
    } else {
        KMM_ASSERT(num_dims <= MEMOPS_MAX_DIMS);

        // The strided kernel handles at most three axes (Rank == 3).
        size_t k = std::min_element(plan.extents, plan.extents + num_dims) - plan.extents;

        CopyPlan sub = plan;
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
            launch_strided_kernel_typed<T>(stream, sub);
        }
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
