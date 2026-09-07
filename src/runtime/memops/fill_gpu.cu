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
#include "kmm/core/point.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/fill_gpu.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm::memops {

struct FillPlan {
    void* dst_addr;
    FillValue fill_pattern;
    size_t line_width;
    size_t num_dims;
    size_t extents[MEMOPS_MAX_DIMS + 1] = {};
    ptrdiff_t strides[MEMOPS_MAX_DIMS + 1] = {};

    template<typename T>
    T pattern_as() const {
        KMM_ASSERT(line_width % sizeof(T) == 0);
        KMM_ASSERT(sizeof(T) % fill_pattern.length == 0);

        std::byte buffer[sizeof(T)];

        for (size_t i = 0; i < sizeof(T); i++) {
            buffer[i] = fill_pattern.buffer[i % fill_pattern.length];
        }

        T value;
        ::memcpy(&value, buffer, sizeof(T));
        return value;
    }

    template<typename T>
    bool is_aligned() const {
        // A T-sized store must cover a whole number of pattern periods.
        if (!is_divisible(sizeof(T), fill_pattern.length)) {
            return false;
        }

        if (!is_divisible(line_width, sizeof(T))) {
            return false;
        }

        if (!is_divisible(reinterpret_cast<uintptr_t>(dst_addr), alignof(T))) {
            return false;
        }

        for (size_t i = 0; i < num_dims; i++) {
            if (!is_divisible(strides[i], alignof(T))) {
                return false;
            }
        }

        return true;
    }
};

FillPlan make_plan(void* dst_base, const FillDescription& description) {
    FillPlan plan;
    plan.num_dims = 0;
    plan.dst_addr = static_cast<std::byte*>(dst_base) + description.offset;
    plan.fill_pattern = description.value;
    plan.line_width = description.value.length;

    for (size_t k : std::array<size_t, 5> {16, 8, 4, 2, 1}) {
        bool is_periodic = plan.fill_pattern.length % k == 0;

        for (size_t i = k; i < plan.fill_pattern.length; i++) {
            is_periodic &= plan.fill_pattern.buffer[i] == plan.fill_pattern.buffer[i - k];
        }

        if (is_periodic) {
            plan.fill_pattern.length = k;
        }
    }

    size_t old_rank = description.num_dims;
    size_t new_rank = 0;
    FillDim dims[MEMOPS_MAX_DIMS] = {};
    std::copy_n(description.dims, old_rank, dims);

    for (size_t i = 0; i < old_rank; i++) {
        for (size_t j = i + 1; j < old_rank; j++) {
            if (unsigned_abs(dims[j].stride) < unsigned_abs(dims[i].stride)) {
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

        // skip this dimension
        if (new_dim.extent == 1) {
            continue;
        }

        // fix negative stride by subtracting offset from the pointer
        if (new_dim.stride < 0) {
            plan.dst_addr =
                static_cast<std::byte*>(plan.dst_addr) + (new_dim.extent - 1) * new_dim.stride;
            new_dim.stride = -new_dim.stride;
        }

        if (new_rank == 0) {
            // if the line width equals the stride, we can just extent the line width.
            if (is_equal(plan.line_width, new_dim.stride)) {
                plan.line_width *= checked_cast<size_t>(new_dim.extent);
                continue;
            }
        } else {
            if (is_equal(new_dim.stride, plan.strides[new_rank - 1] * plan.extents[new_rank - 1])) {
                plan.extents[new_rank - 1] *= new_dim.extent;
                continue;
            }
        }

        plan.extents[new_rank] = new_dim.extent;
        plan.strides[new_rank] = new_dim.stride;
        new_rank++;
    }

    plan.num_dims = new_rank;
    return plan;
}

// Launch `elementwise_fill_kernel` over an index space whose axis 0 is the contiguous run and
// whose remaining axes are strided. The space is peeled apart along each axis so that no single
// launch has more threads than `IndexMapper` can invert (mirrors `ParallelFor::launch_recur`).
template<typename T, size_t Rank>
void launch_strided_fill_recur(
    g_stream_t stream,
    std::byte* dst_addr,
    T value,
    Vec<size_t, Rank> extents,
    Vec<ptrdiff_t, Rank> strides
) {
    constexpr uint32_t max_blocks = 1024;
    constexpr uint32_t threads_per_block = 256;
    Vec<uint32_t, Rank> local_extents;
    uint32_t num_threads = 1;

    for (size_t i = 0; i < Rank; i++) {
        if (extents[i] == 0) {
            return;
        }

        // Largest count along axis `i` that keeps the running thread total within the range
        // that `FastDivisor` accepts as a numerator.
        uint32_t chunk = IndexMapper<uint32_t, Rank>::max_volume / num_threads;

        while (extents[i] > chunk) {
            Vec<size_t, Rank> head = extents;
            head[i] = chunk;
            launch_strided_fill_recur<T, Rank>(stream, dst_addr, value, head, strides);

            dst_addr += ptrdiff_t(chunk) * strides[i];
            extents[i] -= chunk;
        }

        local_extents[i] = static_cast<uint32_t>(extents[i]);  // safe cast
        num_threads *= local_extents[i];
    }

    uint32_t grid_size = std::min(div_ceil(num_threads, threads_per_block), max_blocks);
    auto mapper = IndexMapper<uint32_t, Rank>(local_extents);

    KMM_ASSERT(Rank > 0 && strides[0] == sizeof(T));
    elementwise_fill_kernel<T, Rank>
        <<<grid_size, threads_per_block, 0, stream>>>(dst_addr, value, mapper, strides);
}

template<typename T, size_t Rank>
void launch_strided_kernel_rank_typed(g_stream_t stream, const FillPlan& plan) {
    KMM_ASSERT(plan.num_dims + 1 == Rank);
    Vec<size_t, Rank> extents;
    Vec<ptrdiff_t, Rank> strides;

    extents[0] = plan.line_width / sizeof(T);
    strides[0] = ptrdiff_t(sizeof(T));

    for (size_t i = 1; i < Rank; i++) {
        extents[i] = plan.extents[i - 1];
        strides[i] = plan.strides[i - 1];
    }

    launch_strided_fill_recur<T, Rank>(
        stream,
        static_cast<std::byte*>(plan.dst_addr),
        plan.pattern_as<T>(),
        extents,
        strides
    );
}

template<typename T>
void launch_strided_kernel_typed(g_stream_t stream, const FillPlan& plan) {
    size_t num_dims = plan.num_dims;

    if (num_dims == 0) {
        // a 0D fill is rare since it will typically (always?) be performed by cuMemsetD32Async
        // instead, we turn it into a 1D fill since that will reduce the number of kernels
        // that are precompiled.
        FillPlan p = plan;
        p.extents[0] = 1;
        p.strides[0] = 0;
        p.num_dims++;
        launch_strided_kernel_typed<T>(stream, p);
    } else if (num_dims == 1) {
        launch_strided_kernel_rank_typed<T, 2>(stream, plan);
    } else if (num_dims == 2) {
        launch_strided_kernel_rank_typed<T, 3>(stream, plan);
    } else if (num_dims == 3) {
        launch_strided_kernel_rank_typed<T, 4>(stream, plan);
    } else if (num_dims == 4) {
        launch_strided_kernel_rank_typed<T, 5>(stream, plan);
    } else {
        // should not happen
        KMM_PANIC("invalid dimensionality");
    }
}

bool try_launch_strided_kernel(g_stream_t stream, const FillPlan& plan) {
    if (plan.is_aligned<ulonglong2>()) {
        launch_strided_kernel_typed<ulonglong2>(stream, plan);
    } else if (plan.is_aligned<ulong>()) {
        launch_strided_kernel_typed<ulong>(stream, plan);
    } else if (plan.is_aligned<uint>()) {
        launch_strided_kernel_typed<uint>(stream, plan);
    } else if (plan.is_aligned<ushort>()) {
        launch_strided_kernel_typed<ushort>(stream, plan);
    } else if (plan.is_aligned<std::byte>()) {
        launch_strided_kernel_typed<std::byte>(stream, plan);
    } else {
        return false;
    }

    return true;
}

bool try_fill_plan(g_stream_t stream, const FillPlan& plan) {
    if (plan.line_width == 0) {
        return true;
    }

    if (plan.num_dims == 0) {
        if (plan.is_aligned<uint32_t>()) {
            KMM_GPU_CHECK(g_memset_d32_async(
                reinterpret_cast<g_device_ptr_t>(plan.dst_addr),
                plan.pattern_as<uint32_t>(),
                plan.line_width / sizeof(uint32_t),
                stream
            ));

            return true;
        }

        if (plan.is_aligned<uint16_t>()) {
            KMM_GPU_CHECK(g_memset_d16_async(
                reinterpret_cast<g_device_ptr_t>(plan.dst_addr),
                plan.pattern_as<uint16_t>(),
                plan.line_width / sizeof(uint16_t),
                stream
            ));

            return true;
        }

        if (plan.is_aligned<uint8_t>()) {
            KMM_GPU_CHECK(g_memset_d8_async(
                reinterpret_cast<g_device_ptr_t>(plan.dst_addr),
                plan.pattern_as<uint8_t>(),
                plan.line_width,
                stream
            ));

            return true;
        }
    } else if (plan.num_dims == 1 && is_greater(plan.strides[0], plan.line_width)) {
        if (plan.is_aligned<uint32_t>()) {
            KMM_GPU_CHECK(g_memset_d2d32_async(
                reinterpret_cast<g_device_ptr_t>(plan.dst_addr),
                checked_cast<unsigned int>(plan.strides[0]),
                plan.pattern_as<uint32_t>(),
                plan.line_width / sizeof(uint32_t),
                checked_cast<unsigned int>(plan.extents[0]),
                stream
            ));

            return true;
        }

        if (plan.is_aligned<uint16_t>()) {
            KMM_GPU_CHECK(g_memset_d2d16_async(
                reinterpret_cast<g_device_ptr_t>(plan.dst_addr),
                checked_cast<unsigned int>(plan.strides[0]),
                plan.pattern_as<uint16_t>(),
                plan.line_width / sizeof(uint16_t),
                checked_cast<unsigned int>(plan.extents[0]),
                stream
            ));

            return true;
        }

        if (plan.is_aligned<uint8_t>()) {
            KMM_GPU_CHECK(g_memset_d2d8_async(
                reinterpret_cast<g_device_ptr_t>(plan.dst_addr),
                checked_cast<unsigned int>(plan.strides[0]),
                plan.pattern_as<uint8_t>(),
                plan.line_width,
                checked_cast<unsigned int>(plan.extents[0]),
                stream
            ));

            return true;
        }
    }

    return try_launch_strided_kernel(stream, plan);
}

// Fill a pattern that no single element type can handle: its length is not a power of two
void fill_split_pattern(g_stream_t stream, const FillPlan& plan) {
    size_t pattern_length = plan.fill_pattern.length;
    KMM_ASSERT(pattern_length >= 2);

    size_t offset = 0;
    size_t num_elements = plan.line_width / pattern_length;

    if (num_elements != 1 && plan.num_dims == MEMOPS_MAX_DIMS) {
        throw std::runtime_error("fill plan too complex");
    }

    while (offset < pattern_length) {
        size_t chunk = 32;

        while (true) {
            if (chunk <= pattern_length - offset) {
                FillPlan sub = plan;
                sub.dst_addr = static_cast<std::byte*>(plan.dst_addr) + offset;
                sub.line_width = chunk;
                sub.fill_pattern.length = chunk;
                std::copy_n(plan.fill_pattern.buffer + offset, chunk, sub.fill_pattern.buffer);

                if (num_elements != 1) {
                    sub.num_dims = plan.num_dims + 1;
                    sub.extents[0] = num_elements;
                    sub.strides[0] = checked_cast<ptrdiff_t>(pattern_length);
                    std::copy_n(plan.extents, plan.num_dims, sub.extents + 1);
                    std::copy_n(plan.strides, plan.num_dims, sub.strides + 1);
                }

                if (try_fill_plan(stream, sub)) {
                    offset += chunk;
                    break;
                }
            }

            chunk /= 2;
        }
    }
}

void fill_gpu(g_stream_t stream, void* dst_base, const FillDescription& description) {
    auto plan = make_plan(dst_base, description);

    if (try_fill_plan(stream, plan)) {
        return;
    }

    fill_split_pattern(stream, plan);
}

}  // namespace kmm::memops
