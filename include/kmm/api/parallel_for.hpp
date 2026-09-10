#pragma once

#include <type_traits>
#include <utility>

#include "kmm/api/kernel.hpp"
#include "kmm/core/fast_divisor.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/shape.hpp"

namespace kmm {

namespace detail {
template<typename F, size_t N, typename IndexT, typename... Args>
__global__ void parallel_for_kernel( //
        IndexMapper<uint32_t, N> mapper, F fun, Args... args) {
    uint32_t linear_index = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    Point<N, uint32_t> delta;

    if (mapper.unravel(linear_index, delta)) {
        fun(Point<N, IndexT>::from(delta), args...);
    }
}

template<typename F, size_t N, typename IndexT, typename... Args>
__global__ void parallel_for_offset_kernel(
    Point<N, IndexT> offset,
    IndexMapper<uint32_t, N> mapper,
    F fun,
    Args... args
) {
    uint32_t linear_index = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    Point<N, uint32_t> delta;

    if (mapper.unravel(linear_index, delta)) {
        fun(offset + Point<N, IndexT>::from(delta), args...);
    }
}
}  // namespace detail

/// Launcher (for use with `Device::access`/`DeviceGuard::parallel_for`) that applies `fun` to
/// every point of the N-dimensional index space `shape`.
template<typename F, size_t N>
class ParallelFor {
  public:
    using index_type = default_index_type;

    explicit ParallelFor(Shape<N, index_type> shape, F fun, unsigned int block_size = 256) :
        m_shape(shape),
        m_fun(std::move(fun)),
        m_block_size(block_size) {}

    template<typename... Args>
    void operator()(g_stream_t stream, Args&&... args) const {
        launch_recur(stream, m_offset, m_shape, args...);
    }

  private:
    template<typename... Args>
    void launch_recur(
        g_stream_t stream,
        Point<N, index_type> offset,
        Shape<N, index_type> shape,
        const Args&... args
    ) const {
        if (shape.is_empty()) {
            return;
        }

        uint32_t num_threads = 1;

        for (size_t i = 0; i < N; i++) {
            // chunk is the maximum number of elements that can be along the i-th
            // dimensions without `num_threads` exceeding its maximum.
            auto chunk = index_type(IndexMapper<uint32_t, N>::max_volume / num_threads);

            // if chunk < shape[i], then we must split the i-th dimensions.
            // we strip off the first `chunk` elements and launch it recursively.
            while (is_less(chunk, shape[i])) {
                Shape<N> head = shape;
                head[i] = chunk;
                launch_recur(stream, offset, head, args...);

                offset[i] += chunk;
                shape[i] -= chunk;
            }

            num_threads *= uint32_t(shape[i]);
        }

        uint32_t grid_size = div_ceil(num_threads, m_block_size);
        auto mapper = IndexMapper<uint32_t, N>(Shape<N, uint32_t>::from(shape));

        if (offset == Point<N>::zero()) {
            detail::parallel_for_kernel<F, N, index_type, std::decay_t<Args>...>
                <<<grid_size, m_block_size, 0, stream>>>(mapper, m_fun, args...);
        } else {
            detail::parallel_for_offset_kernel<F, N, index_type, std::decay_t<Args>...>
                <<<grid_size, m_block_size, 0, stream>>>(offset, mapper, m_fun, args...);
        }
    }

    F m_fun;
    Point<N, index_type> m_offset;
    Shape<N, index_type> m_shape;
    uint32_t m_block_size;
};

}  // namespace kmm
