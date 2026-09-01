#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/fill_gpu.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm::memops {

template<typename T, size_t Rank>
void __global__ strided_fill_kernel(void* dst_addr, T fill_value, Vec<FillDim, Rank> dims) {
    static_assert(Rank <= 3, "invalid rank");

    uint p[3] = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    };

    for (size_t i = 0; i < Rank; i++) {
        if (p[i] >= dims[i].extent) {
            return;
        }

        dst_addr = static_cast<std::byte*>(dst_addr) + ptrdiff_t(p[i]) * dims[i].stride;
    }

    *static_cast<T*>(dst_addr) = fill_value;
}

template<typename T, size_t Rank>
void launch_strided_fill(g_stream_t stream, void* dst_addr, T fill_value, const FillDim* src_dims) {
    KMM_ASSERT(reinterpret_cast<uintptr_t>(dst_addr) % alignof(T) == 0);

    // `simplify()` orders `src_dims` from the largest stride to the smallest, so `src_dims`'s
    // last axis is the innermost (most contiguous) one. Reverse the axes into `dims` so the
    // kernel's fastest thread index (x) walks that innermost axis: neighbouring threads then
    // write neighbouring addresses and the stores coalesce.
    Vec<FillDim, Rank> dims;
    for (size_t i = 0; i < Rank; i++) {
        dims[i] = src_dims[Rank - 1 - i];

        // Every strided position must land on an `alignof(T)` boundary so the kernel's store is
        // aligned.
        KMM_ASSERT(dims[i].stride % memops_stride_type(alignof(T)) == 0);
    }

    // Extent of each kernel axis after the reversal (absent axes contribute a single iteration).
    auto axis_extent = [&](size_t axis) -> uint {
        return axis < Rank ? checked_cast<uint>(dims[axis].extent) : 1u;
    };

    // Spend the ~256-thread budget on the innermost axis first (rounded to a warp so it stays
    // coalesced), then hand any surplus to the next axis so short inner rows still fill a block.
    uint warp = 32;
    uint inner = std::min(round_up_to_multiple(std::min(axis_extent(0), 256u), warp), 256u);
    dim3 block_size(inner, 256u / inner, 1);

    dim3 grid_size(
        div_ceil(axis_extent(0), block_size.x),
        div_ceil(axis_extent(1), block_size.y),
        div_ceil(axis_extent(2), block_size.z)
    );

    strided_fill_kernel<T, Rank><<<grid_size, block_size, 0, stream>>>(dst_addr, fill_value, dims);
}

template<typename T>
void launch_strided_fill_rank(
    g_stream_t stream,
    void* dst_addr,
    const FillDescription& description
) {
    static constexpr FillDim unit_dim = {1, 0};

    KMM_ASSERT(description.value.length == sizeof(T));
    T fill_value;
    ::memcpy(&fill_value, description.value.buffer, sizeof(T));

    switch (description.num_dims) {
        case 0:
            // forward to Rank=1 instead of Rank=0. The case Rank=0 is so rare that is not worth it to
            // specialize on that particular case and precompile many kernels that are rarely called.
            return launch_strided_fill<T, 1>(stream, dst_addr, fill_value, &unit_dim);
        case 1:
            return launch_strided_fill<T, 1>(stream, dst_addr, fill_value, description.dims);
        case 2:
            return launch_strided_fill<T, 2>(stream, dst_addr, fill_value, description.dims);
        case 3:
            return launch_strided_fill<T, 3>(stream, dst_addr, fill_value, description.dims);
    }

    KMM_PANIC("fill_gpu: unsupported number of dimensions");
}

// Returns true if the first `period` bytes of `value` tile its entire length, i.e. the value is
// `value.length / period` back-to-back copies of that `period`-byte pattern. A `period` of zero,
// or one that does not divide `value.length`, is never a valid tiling.
bool is_repeating(const FillValue& value, size_t period) {
    if (period == 0 || value.length % period != 0) {
        return false;
    }

    for (size_t i = period; i < value.length; i++) {
        if (value.buffer[i] != value.buffer[i - period]) {
            return false;
        }
    }

    return true;
}

// attempt to rewrite the given fill description to instead use the given data type T.
template<typename T>
std::optional<FillDescription> retype_fill_as(void* dst_addr, const FillDescription& description) {
    constexpr size_t word = sizeof(T);

    const size_t value_length = description.value.length;
    const size_t num_dims = description.num_dims;

    if (value_length == 0) {
        return std::nullopt;
    }

    // The address must be aligned to T.
    if (reinterpret_cast<uintptr_t>(dst_addr) % alignof(T) != 0) {
        return std::nullopt;
    }

    // Each stride must be aligned to T
    for (size_t i = 0; i < num_dims; i++) {
        if (!is_divisible(description.dims[i].stride, alignof(T))) {
            return std::nullopt;
        }
    }

    // We attempt to store T multiple times. Either:
    // 1) the fill value is a whole number of `word`-byte tiles (so one tile can stand in for it),
    // 2) `word` is a multiple of the fill value length (so the value tiles a single `T`).
    if (word < value_length) {
        if (!is_repeating(description.value, word)) {
            return std::nullopt;
        }
    } else {
        if (word % value_length != 0) {
            return std::nullopt;
        }
    }

    FillDescription result;
    result.offset = description.offset;
    result.num_dims = num_dims;
    std::copy_n(description.dims, num_dims, result.dims);

    // The replacement value is one `word`-wide slice of the (tiled) pattern.
    result.value.length = word;
    for (size_t i = 0; i < word; i++) {
        result.value.buffer[i] = description.value.buffer[i % value_length];
    }

    // attempt to repeat the last dimension if possible
    if (num_dims != 0) {
        FillDim& inner = result.dims[num_dims - 1];
        memops_stride_type line_width = checked_mul<memops_stride_type>(inner.extent, value_length);

        if (is_equal(inner.stride, value_length) && is_divisible(line_width, word)) {
            inner.extent = checked_div<memops_extent_type>(line_width, word);
            inner.stride = checked_cast<memops_stride_type>(word);
            return result;
        }
    }

    if (word != value_length) {
        if (word > value_length || num_dims >= 3) {
            std::cout << "invalid copy" << std::endl;
            std::cout << "- value_length: " << description.value.length << std::endl;

            for (size_t i = 0; i < num_dims; i++) {
                std::cout << " - dim: extent=" << description.dims[i].extent
                          << ", stride=" << description.dims[i].stride << std::endl;
            }

            return std::nullopt;
        }

        result.dims[num_dims] =
            FillDim {memops_extent_type(value_length / word), memops_stride_type(word)};
        result.num_dims += 1;
    }

    return result;
}

struct uint64x2_t {
    uint64_t x, y;
};
struct uint32x2_t {
    uint32_t x, y;
};

// Attempts to carry out `description` with a single strided-fill kernel launch, reinterpreting
// the fill value as the widest POD type whose alignment `dst_addr` and every stride satisfy.
// Returns false if no supported type fits, in which case the caller must split the fill value
// into smaller (more weakly aligned) pieces.
bool try_strided_fill(g_stream_t stream, void* dst_addr, const FillDescription& description) {
    if (description.value.length == 0) {
        return true;
    }

    // 128 bit
    if (auto plan = retype_fill_as<ulonglong2>(dst_addr, description)) {
        launch_strided_fill_rank<ulonglong2>(stream, dst_addr, *plan);
        return true;
    }

    // 128 bit, 64 bit alignment
    if (auto plan = retype_fill_as<uint64x2_t>(dst_addr, description)) {
        launch_strided_fill_rank<uint64x2_t>(stream, dst_addr, *plan);
        return true;
    }

    // 64 bit
    if (auto plan = retype_fill_as<uint64_t>(dst_addr, description)) {
        launch_strided_fill_rank<uint64_t>(stream, dst_addr, *plan);
        return true;
    }

    // 64 bit, 32 bit alignment
    if (auto plan = retype_fill_as<uint32x2_t>(dst_addr, description)) {
        launch_strided_fill_rank<uint32x2_t>(stream, dst_addr, *plan);
        return true;
    }

    // 32 bit
    if (auto plan = retype_fill_as<uint32_t>(dst_addr, description)) {
        launch_strided_fill_rank<uint32_t>(stream, dst_addr, *plan);
        return true;
    }

    // 16 bit
    if (auto plan = retype_fill_as<uint16_t>(dst_addr, description)) {
        launch_strided_fill_rank<uint16_t>(stream, dst_addr, *plan);
        return true;
    }

    // 8 bit
    if (auto plan = retype_fill_as<uint8_t>(dst_addr, description)) {
        launch_strided_fill_rank<uint8_t>(stream, dst_addr, *plan);
        return true;
    }

    return false;
}

// Parameters for a contiguous fill. When `height == 1` the fill is a single 1D run of `width`
// elements and `pitch` is unused; otherwise it is `height` rows of `width` elements each, with
// consecutive rows `pitch` bytes apart.
struct Fill2D {
    size_t width;  // number of elements in each row
    size_t height;  // number of rows (1 for a plain 1D fill)
    size_t pitch;  // distance in bytes between consecutive rows (unused when height == 1)
};

// Returns the fill parameters if `description` can be carried out by a single memset, 1D or 2D.
template<typename T>
std::optional<Fill2D> contiguous_fill_2d(const void* dst_addr, const FillDescription& description) {
    size_t fill_width = sizeof(T);
    size_t value_width = description.value.length;

    if (!is_repeating(description.value, fill_width)) {
        return std::nullopt;
    }

    if (reinterpret_cast<uintptr_t>(dst_addr) % fill_width != 0) {
        return std::nullopt;
    }

    // `simplify()` orders `dims` from the largest stride to the smallest, so the outer axis is
    // the row axis and (for a 2D description) the inner axis is the contiguous run within a row.
    size_t row_bytes;
    memops_extent_type height;
    memops_stride_type pitch;

    if (description.num_dims == 0) {
        row_bytes = value_width;
        height = 1;
        pitch = 0;
    } else if (description.num_dims == 1) {
        row_bytes = value_width;
        height = description.dims[0].extent;
        pitch = description.dims[0].stride;
    } else if (description.num_dims == 2 && is_equal(description.dims[1].stride, value_width)) {
        row_bytes = checked_mul<size_t>(description.dims[1].extent, value_width);
        height = description.dims[0].extent;
        pitch = description.dims[0].stride;
    } else {
        // 3+ dims, or a 2D fill whose inner axis is itself strided: not expressible as a single
        // memset, so leave it to the strided-fill kernel.
        return std::nullopt;
    }

    if (!is_divisible(row_bytes, fill_width)) {
        return std::nullopt;
    }

    // A single row (or an empty fill) is one contiguous run: emit a 1D fill.
    if (height <= 1) {
        return Fill2D {height == 0 ? size_t {0} : row_bytes / fill_width, 1, 0};
    }

    // Rows laid out back-to-back are also one contiguous run, so collapse them into a 1D fill.
    // `row_bytes` is a multiple of `fill_width` (checked above); divide before multiplying so the
    // element count cannot overflow on a huge fill.
    if (is_equal(pitch, row_bytes)) {
        return Fill2D {checked_mul<size_t>(row_bytes / fill_width, height), 1, 0};
    }

    if (pitch <= 0 || !is_divisible(pitch, fill_width)) {
        return std::nullopt;
    }

    return Fill2D {
        row_bytes / fill_width,
        checked_cast<size_t>(height),
        checked_cast<size_t>(pitch),
    };
}

// Carries out an already-simplified fill at an absolute address, trying progressively less
// specialised strategies in turn.
static void dispatch_fill(g_stream_t stream, void* dst_addr, const FillDescription& description) {
    if (auto f = contiguous_fill_2d<unsigned int>(dst_addr, description)) {
        unsigned int value;
        ::memcpy(&value, description.value.buffer, sizeof(unsigned int));

        if (f->height == 1) {
            KMM_GPU_CHECK(g_memset_d32_async((g_device_ptr_t)dst_addr, value, f->width, stream));
        } else {
            KMM_GPU_CHECK(g_memset_2d_d32_async(
                (g_device_ptr_t)dst_addr,
                f->pitch,
                value,
                f->width,
                f->height,
                stream
            ));
        }
        return;
    }

    if (auto f = contiguous_fill_2d<unsigned short>(dst_addr, description)) {
        unsigned short value;
        ::memcpy(&value, description.value.buffer, sizeof(unsigned short));

        if (f->height == 1) {
            KMM_GPU_CHECK(g_memset_d16_async((g_device_ptr_t)dst_addr, value, f->width, stream));
        } else {
            KMM_GPU_CHECK(g_memset_2d_d16_async(
                (g_device_ptr_t)dst_addr,
                f->pitch,
                value,
                f->width,
                f->height,
                stream
            ));
        }
        return;
    }

    if (auto f = contiguous_fill_2d<unsigned char>(dst_addr, description)) {
        unsigned char value;
        ::memcpy(&value, description.value.buffer, sizeof(unsigned char));

        if (f->height == 1) {
            KMM_GPU_CHECK(g_memset_d8_async((g_device_ptr_t)dst_addr, value, f->width, stream));
        } else {
            KMM_GPU_CHECK(g_memset_2d_d8_async(
                (g_device_ptr_t)dst_addr,
                f->pitch,
                value,
                f->width,
                f->height,
                stream
            ));
        }
        return;
    }

    if (try_strided_fill(stream, dst_addr, description)) {
        return;
    }

    // Nothing above could handle the fill value at this alignment. Split it into two halves and
    // fill each recursively: each half is smaller (so more weakly aligned) and is retried against
    // every strategy above, not just the strided kernel.
    size_t left_size = round_up_to_power_of_two(description.value.length) / 2;
    size_t right_size = description.value.length - left_size;
    KMM_ASSERT(left_size > 0 && right_size > 0);

    auto spec = description;
    spec.value.length = left_size;
    std::copy_n(description.value.buffer, left_size, spec.value.buffer);
    dispatch_fill(stream, dst_addr, spec);

    spec.value.length = right_size;
    std::copy_n(description.value.buffer + left_size, right_size, spec.value.buffer);
    dispatch_fill(stream, static_cast<std::byte*>(dst_addr) + left_size, spec);
}

void fill_gpu(g_stream_t stream, void* dst_base, const FillDescription& description) {
    FillDescription simplified = description.simplify();
    void* dst_addr = static_cast<std::byte*>(dst_base) + simplified.offset;
    dispatch_fill(stream, dst_addr, simplified);
}

}  // namespace kmm::memops
