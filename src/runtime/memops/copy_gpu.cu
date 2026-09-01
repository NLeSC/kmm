#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "kmm/core/integer_fun.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/runtime/memops/copy_gpu.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm::memops {

template<typename T, size_t Rank>
__global__ void strided_copy_kernel(const void* src_ptr, void* dst_ptr, Vec<CopyDim, Rank> dims) {
    static_assert(Rank <= 3, "invalid rank");

    uint p[3] = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    };

    const std::byte* src = static_cast<const std::byte*>(src_ptr);
    std::byte* dst = static_cast<std::byte*>(dst_ptr);

    for (size_t i = 0; i < Rank; i++) {
        if (p[i] >= dims[i].extent) {
            return;
        }

        src += ptrdiff_t(p[i]) * dims[i].src_stride;
        dst += ptrdiff_t(p[i]) * dims[i].dst_stride;
    }

    const T* __restrict__ s = reinterpret_cast<const T*>(src);
    T* __restrict__ d = reinterpret_cast<T*>(dst);
    *d = *s;
}

template<typename T, size_t Rank>
void launch_strided_copy(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    Vec<CopyDim, Rank> dims
) {
    KMM_ASSERT(reinterpret_cast<uintptr_t>(src_addr) % alignof(T) == 0);
    KMM_ASSERT(reinterpret_cast<uintptr_t>(dst_addr) % alignof(T) == 0);

    // The kernel maps every axis onto a 32-bit grid coordinate, so an axis whose extent does not
    // fit in `uint` is peeled into `uint`-sized chunks and copied by recursion.
    constexpr memops_extent_type max_extent = std::numeric_limits<uint>::max();

    for (size_t i = 0; i < Rank; i++) {
        KMM_ASSERT(is_divisible(dims[i].src_stride, alignof(T)));
        KMM_ASSERT(is_divisible(dims[i].dst_stride, alignof(T)));

        if (dims[i].extent > max_extent) {
            for (memops_extent_type offset = 0; offset < dims[i].extent; offset += max_extent) {
                memops_extent_type chunk = std::min(max_extent, dims[i].extent - offset);

                Vec<CopyDim, Rank> chunk_dims = dims;
                chunk_dims[i].extent = chunk;

                launch_strided_copy<T, Rank>(
                    stream,
                    static_cast<const std::byte*>(src_addr) + offset * dims[i].src_stride,
                    static_cast<std::byte*>(dst_addr) + offset * dims[i].dst_stride,
                    chunk_dims
                );
            }

            return;
        }
    }

    // Extent of each kernel axis (absent axes contribute a single iteration).
    auto axis_extent = [&](size_t axis) -> uint {
        return axis < Rank ? uint(dims[axis].extent) : 1u;
    };

    uint warp_size = 32;
    uint threads_per_block = 256;
    uint inner_extent = axis_extent(0);
    uint block_size_x;

    if (inner_extent < warp_size) {
        block_size_x = round_up_to_power_of_two(inner_extent);
    } else if (inner_extent < threads_per_block) {
        block_size_x = round_up_to_multiple(inner_extent, warp_size);
    } else {
        block_size_x = threads_per_block;
    }

    dim3 block_size = {block_size_x, threads_per_block / block_size_x, 1};
    dim3 grid_size = {
        div_ceil(axis_extent(0), block_size.x),
        div_ceil(axis_extent(1), block_size.y),
        div_ceil(axis_extent(2), block_size.z),
    };

    strided_copy_kernel<T, Rank><<<grid_size, block_size, 0, stream>>>(src_addr, dst_addr, dims);
}

template<typename T>
void copy_typed_async(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
) {
    KMM_ASSERT(is_divisible(description.element_size, sizeof(T)));
    size_t repeat = description.element_size / sizeof(T);

    // The first dimension always just repeating T with contiguous stride
    CopyDim unit_dim = CopyDim {checked_cast<memops_extent_type>(repeat), sizeof(T), sizeof(T)};

    switch (description.num_dims) {
        case 1: {
            auto dims = Vec(unit_dim, description.dims[0]);
            return launch_strided_copy<T, 2>(stream, src_addr, dst_addr, dims);
        }
        case 2: {
            auto dims = Vec(unit_dim, description.dims[1], description.dims[0]);
            return launch_strided_copy<T, 3>(stream, src_addr, dst_addr, dims);
        }
    }

    KMM_PANIC("copy_gpu: unsupported number of dimensions");
}

template<typename T>
bool is_copy_aligned_for(const void* src_addr, void* dst_addr, const CopyDescription& description) {
    const size_t element_size = description.element_size;
    const size_t num_dims = description.num_dims;

    // must be aligned to T
    if (reinterpret_cast<uintptr_t>(dst_addr) % alignof(T) != 0
        || reinterpret_cast<uintptr_t>(src_addr) % alignof(T) != 0) {
        return false;
    }

    // Each stride must be aligned to T
    for (size_t i = 0; i < num_dims; i++) {
        if (!is_divisible(description.dims[i].dst_stride, alignof(T))
            || !is_divisible(description.dims[i].src_stride, alignof(T))) {
            return false;
        }
    }

    // element size must also be divisible by T
    if (!is_divisible(element_size, sizeof(T))) {
        return false;
    }

    return true;
}

void dispatch_copy_gpu(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
);

// Handles a copy with more axes than the kernel can map: peel off the shortest axis (fewest
// recursive launches) and dispatch each of its slices as an independent `(rank - 1)`-dimensional
// copy. `description` must have at least one axis.
void split_copy_gpu(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
) {
    size_t smallest_axis = 0;
    for (size_t i = 1; i < description.num_dims; i++) {
        if (description.dims[i].extent < description.dims[smallest_axis].extent) {
            smallest_axis = i;
        }
    }

    const CopyDim axis = description.dims[smallest_axis];

    CopyDescription smaller(description.element_size);
    smaller.num_dims = description.num_dims - 1;
    for (size_t j = 0, k = 0; j < description.num_dims; j++) {
        if (j != smallest_axis) {
            smaller.dims[k++] = description.dims[j];
        }
    }

    for (memops_extent_type i = 0; i < axis.extent; i++) {
        dispatch_copy_gpu(
            stream,
            static_cast<const std::byte*>(src_addr) + axis.src_stride * i,
            static_cast<std::byte*>(dst_addr) + axis.dst_stride * i,
            smaller
        );
    }
}

void dispatch_copy_gpu(
    g_stream_t stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
) {
    size_t rank = description.num_dims;

    // For rank 0, this is just one big contiguous copy.
    if (rank == 0) {
        KMM_GPU_CHECK(g_memcpy_d_to_d_async(
            (g_device_ptr_t)src_addr,
            (g_device_ptr_t)dst_addr,
            description.element_size,
            stream
        ));

        return;
    }

    // For rank 1 and 2, we can launch a strided copy kernel of type T.
    if (rank <= 2) {
        if (is_copy_aligned_for<ulonglong2>(src_addr, dst_addr, description)) {
            copy_typed_async<ulonglong2>(stream, src_addr, dst_addr, description);
            return;
        }

        if (is_copy_aligned_for<uint64_t>(src_addr, dst_addr, description)) {
            copy_typed_async<uint64_t>(stream, src_addr, dst_addr, description);
            return;
        }

        if (is_copy_aligned_for<uint32_t>(src_addr, dst_addr, description)) {
            copy_typed_async<uint32_t>(stream, src_addr, dst_addr, description);
            return;
        }

        if (is_copy_aligned_for<uint16_t>(src_addr, dst_addr, description)) {
            copy_typed_async<uint16_t>(stream, src_addr, dst_addr, description);
            return;
        }

        // Just a contiguous copy of `element_size` bytes
        copy_typed_async<uint8_t>(stream, src_addr, dst_addr, description);
    }

    // for rank > 2, we must split the extent of one axis.
    split_copy_gpu(stream, src_addr, dst_addr, description);
}

void copy_gpu(
    g_stream_t stream,
    const void* src_base,
    void* dst_base,
    const CopyDescription& description
) {
    CopyDescription simplified = description.simplify();
    const void* src_addr = static_cast<const std::byte*>(src_base) + simplified.src_offset;
    void* dst_addr = static_cast<std::byte*>(dst_base) + simplified.dst_offset;

    dispatch_copy_gpu(stream, src_addr, dst_addr, simplified);
}

}  // namespace kmm::memops
