#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kmm/core/fast_divisor.hpp"
#include "kmm/core/vec.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm::memops {

//===========================================================================
// Copy kernels (see copy_gpu.cu)
//===========================================================================

template<typename T, size_t Rank, uint32_t TileSize, uint32_t BlockSizeX, uint32_t BlockSizeY>
__global__ void transpose_copy_kernel(
    std::byte* dst_addr,
    const std::byte* src_addr,
    uint32_t extent0,
    uint32_t extent1,
    IndexMapper<uint32_t, Rank> mapper,
    Vec<ptrdiff_t, Rank> dst_strides,
    Vec<ptrdiff_t, Rank> src_strides
) {
    static_assert(TileSize % BlockSizeX == 0, "invalid block size x");
    static_assert(TileSize % BlockSizeY == 0, "invalid block size y");

    __shared__ T shared_tile[TileSize][TileSize + 1];
    uint32_t linear_index = blockIdx.x;
    Vec<uint32_t, Rank> p;

    while (mapper.unravel(linear_index, p)) {
        std::byte* dst = dst_addr;
        const std::byte* src = src_addr;

        // skip axis 0 and 1 since they are tiled.
        for (size_t i = 2; i < Rank; i++) {
            dst += dst_strides[i] * ptrdiff_t(p[i]);
            src += src_strides[i] * ptrdiff_t(p[i]);
        }

        // Load a TileSize x TileSize tile into shared memory. Consecutive threads along x
        // read consecutive elements along axis 0, keeping the source access coalesced.
#pragma unroll
        for (uint32_t x = 0; x < TileSize; x += BlockSizeX) {
#pragma unroll
            for (uint32_t y = 0; y < TileSize; y += BlockSizeY) {
                uint32_t tx = threadIdx.x + x;
                uint32_t ty = threadIdx.y + y;
                uint32_t px = p[0] * TileSize + tx;
                uint32_t py = p[1] * TileSize + ty;
                T value {};

                if (px < extent0 && py < extent1) {
                    value = *reinterpret_cast<const T*>(
                        src + ptrdiff_t(px) * src_strides[0] + ptrdiff_t(py) * src_strides[1]
                    );
                }

                shared_tile[ty][tx] = value;
            }
        }

        __syncthreads();

        // Write the tile back transposed. Consecutive threads along x now write consecutive
        // elements along axis 1, keeping the destination access coalesced.
#pragma unroll
        for (uint32_t x = 0; x < TileSize; x += BlockSizeX) {
#pragma unroll
            for (uint32_t y = 0; y < TileSize; y += BlockSizeY) {
                uint32_t tx = threadIdx.x + x;
                uint32_t ty = threadIdx.y + y;
                uint32_t px = p[0] * TileSize + ty;
                uint32_t py = p[1] * TileSize + tx;

                if (px < extent0 && py < extent1) {
                    *reinterpret_cast<T*>(
                        dst + ptrdiff_t(px) * dst_strides[0] + ptrdiff_t(py) * dst_strides[1]
                    ) = shared_tile[tx][ty];
                }
            }
        }

        __syncthreads();

        linear_index += gridDim.x;
    }
}

template<typename T, size_t Rank>
__global__ void elementwise_copy_kernel(
    std::byte* dst_addr,
    const std::byte* src_addr,
    IndexMapper<uint32_t, Rank> mapper,
    Vec<ptrdiff_t, Rank> dst_strides,
    Vec<ptrdiff_t, Rank> src_strides
) {
    uint32_t linear_index = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    Vec<uint32_t, Rank> p;

    while (mapper.unravel(linear_index, p)) {
        std::byte* dst = dst_addr;
        const std::byte* src = src_addr;

        dst += ptrdiff_t(sizeof(T)) * ptrdiff_t(p[0]);
        src += ptrdiff_t(sizeof(T)) * ptrdiff_t(p[0]);

#pragma unroll
        for (size_t i = 1; i < Rank; i++) {
            dst += dst_strides[i] * ptrdiff_t(p[i]);
            src += src_strides[i] * ptrdiff_t(p[i]);
        }

        *reinterpret_cast<T*>(dst) = *reinterpret_cast<const T*>(src);
        linear_index += blockDim.x * gridDim.x;
    }
}

//===========================================================================
// Fill kernels (see fill_gpu.cu)
//===========================================================================

template<typename T, size_t Rank>
__global__ void elementwise_fill_kernel(
    std::byte* dst_addr,
    T value,
    IndexMapper<uint32_t, Rank> mapper,
    Vec<ptrdiff_t, Rank> strides
) {
    uint32_t linear_index = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    Vec<uint32_t, Rank> p;

    while (mapper.unravel(linear_index, p)) {
        std::byte* addr = dst_addr;

        // strides[0] == sizeof(T)
        addr += ptrdiff_t(sizeof(T)) * ptrdiff_t(p[0]);

#pragma unroll
        for (size_t i = 1; i < Rank; i++) {
            addr += strides[i] * ptrdiff_t(p[i]);
        }

        *reinterpret_cast<T*>(addr) = value;
        linear_index += blockDim.x * gridDim.x;
    }
}

//===========================================================================
// Reduction kernels (see reduction_gpu.cu)
//===========================================================================

// Wavefront/warp width for CUDA and HIP
#if defined(KMM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
constexpr uint32_t KMM_REDUCE_WARP_SIZE = __AMDGCN_WAVEFRONT_SIZE__;
#else
constexpr uint32_t KMM_REDUCE_WARP_SIZE = 32;
#endif

// Generic warp shuffle that works for any trivially-copyable `T` (not just the scalar types that
// `__shfl_xor_sync` natively overloads, such as `KeyValue<...>`), by shuffling it word-by-word.
template<typename T>
KMM_DEVICE T shfl_xor(T value, int offset) {
    static_assert(sizeof(T) % sizeof(uint32_t) == 0, "size of T must be a multiple of 4 bytes");
    uint32_t words[sizeof(T) / sizeof(uint32_t)];
    std::memcpy(words, &value, sizeof(T));

#pragma unroll
    for (auto& word : words) {
#if defined(KMM_USE_HIP)
        word = __shfl_xor(word, offset);
#else
        word = __shfl_xor_sync(0xffffffffu, word, offset);
#endif
    }

    std::memcpy(&value, words, sizeof(T));
    return value;
}

/// Performs `dst_addr[I] += src_addr[I]` for each index I in `mapper`.
template<typename Reduction, size_t N, uint32_t BlockSize>
__global__ void elementwise_fold_kernel(
    const std::byte* __restrict__ src_addr,
    std::byte* __restrict__ dst_addr,
    IndexMapper<uint32_t, N> mapper,
    Vec<ptrdiff_t, N> input_strides,
    Vec<ptrdiff_t, N> output_strides
) {
    using T = typename Reduction::element_type;
    uint32_t tx = threadIdx.x;
    uint32_t linear_index = uint32_t(blockIdx.x) * BlockSize + tx;
    Vec<uint32_t, N> p;

    while (mapper.unravel(linear_index, p)) {
        const auto* src = src_addr;
        auto* dst = dst_addr;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            src += input_strides[i] * ptrdiff_t(p[i]);
            dst += output_strides[i] * ptrdiff_t(p[i]);
        }

        Reduction accum = Reduction {*reinterpret_cast<T*>(dst)};

        T value = *reinterpret_cast<const T*>(src);
        accum.consume(value);

        *reinterpret_cast<T*>(dst) = accum.finish();
        linear_index += BlockSize * gridDim.x;
    }
}

/// Indicates what kind of reduction must be performed. In all cases:
/// * blockDim.x: number of outputs produced by one block.
/// * blockDim.y: number of threads that cooperate to produce that one output.
enum class ReductionKernelKind {
    // kernel is launched with thread block dimensions (1, BlockSize). Each thread is given a unique output and
    // will perform the full reduction for that one output.
    Elementwise,

    // kernel is launched with thread block dimensions (WARP_SIZE, BlockSize/WARP_SIZE). Each thread in a warp
    // is given a different output, but there are multiple warps that will work together to perform the reduction.
    Warpwise,

    // kernel is launched with thread block dimensions (BlockSize, 1). All threads are given the same output and they
    // must all work together to perform the reduction.
    Blockwise
};

template<typename Reduction, size_t N, uint32_t BlockSize>
__global__ void elementwise_reduce_kernel(
    const std::byte* __restrict__ src_addr,
    std::byte* __restrict__ dst_addr,
    bool accumulate,
    uint32_t reduction_extent,
    ptrdiff_t reduction_stride,
    IndexMapper<uint32_t, N> mapper,
    Vec<ptrdiff_t, N> input_strides,
    Vec<ptrdiff_t, N> output_strides,
    ReductionKernelKind kind
) {
    using T = typename Reduction::element_type;
    uint32_t tx = threadIdx.x;
    __shared__ T shared_values[2][BlockSize / 2];

    uint32_t linear_index = uint32_t(blockIdx.x) * blockDim.x + tx;
    Vec<uint32_t, N> p {};

    for (;; linear_index += blockDim.x * gridDim.x) {
        // from some reason, using `int` instead of `bool` leads to much cleaner ptx.
        int active = mapper.unravel(linear_index, p);

        // only if all threads in this block want to exit, we actually exit. This prevents deadlock on the
        // upcoming __syncthreads as all threads must remain active for that to work.
        if (!__syncthreads_or(active)) {
            break;
        }

        const auto* src = src_addr + ptrdiff_t(threadIdx.y) * reduction_stride;
        auto* dst = dst_addr;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            src += input_strides[i] * ptrdiff_t(p[i]);
            dst += output_strides[i] * ptrdiff_t(p[i]);
        }

        Reduction accum {};

        if (active) {
            for (uint32_t current = threadIdx.y; current < reduction_extent;
                 current += blockDim.y) {
                T value = *reinterpret_cast<const T*>(src);
                accum.consume(value);
                src += reduction_stride * blockDim.y;
            }
        }

        // if not elementwise (i.e., one thread per output), then we must do a reduction
        if (kind != ReductionKernelKind::Elementwise) {
            bool parity = false;
            uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;

            // reduce the values across the block into a single warp.
#pragma unroll
            for (uint32_t stride = BlockSize / 2; stride >= KMM_REDUCE_WARP_SIZE; stride /= 2) {
                parity = !parity;

                // threads that are active and tid >= stride (or alternatively, having NOT tid < stride),
                // will write there value and then deactivate themselves.
                if (active && tid >= stride) {
                    // the __syncthreads_or protects the first write, the successive writes will have
                    // __syncthreads from each iteration to protect shared memory.
                    shared_values[parity][tid - stride] = accum.finish();
                    active = false;
                }

                __syncthreads();

                if (active) {
                    // the above __syncthreads ensures that all values have been written.
                    accum.consume(shared_values[parity][tid]);
                }
            }

            // if blockwise, we must reduce the warp to a single value.
            if (kind == ReductionKernelKind::Blockwise) {
#pragma unroll
                for (uint32_t offset = KMM_REDUCE_WARP_SIZE / 2; offset >= 1; offset /= 2) {
                    accum.consume(shfl_xor(accum.finish(), int(offset)));
                }
            }

            // Only threads in the first warp remain active
            active &= threadIdx.y == 0;
        }

        if (active) {
            if (accumulate) {
                accum.consume(*reinterpret_cast<T*>(dst));
            }

            *reinterpret_cast<T*>(dst) = accum.finish();
        }
    }
}

}  // namespace kmm::memops
