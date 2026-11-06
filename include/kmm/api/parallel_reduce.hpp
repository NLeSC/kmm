#pragma once

#include <type_traits>
#include <utility>

#ifdef KMM_USE_CUDA
    #include "cub/block/block_reduce.cuh"
#elif KMM_USE_HIP
    #include "hipcub/hipcub.hpp"
namespace cub = hipcub;
#endif

#include "kmm/api/device.hpp"
#include "kmm/api/kernel.hpp"
#include "kmm/core/fast_divisor.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/shape.hpp"

namespace kmm {

namespace detail {
template<typename F, size_t N, typename IndexT, typename OutputT, typename... Args>
__global__ void parallel_reduce_kernel( //
        IndexMapper<uint32_t, N> mapper, F fun, OutputT* output_addr, Args... args) {
    uint32_t linear_index = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    Point<N, uint32_t> delta;
    OutputT output {};

    if (mapper.unravel(linear_index, delta)) {
        output += fun(Point<N, IndexT>::from(delta), args...);
    }

    output = cub::BlockReduce<OutputT, 256>().Sum(output);

    if (threadIdx.x == 0) {
        output_addr[blockIdx.x] = output;
    }
}

template<typename F, size_t N, typename IndexT, typename OutputT, typename... Args>
__global__ void parallel_reduce_offset_kernel(
    Point<N, IndexT> offset,
    IndexMapper<uint32_t, N> mapper,
    F fun,
    OutputT* output_addr,
    Args... args
) {
    uint32_t linear_index = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    Point<N, uint32_t> delta;
    OutputT output {};

    if (mapper.unravel(linear_index, delta)) {
        output += fun(offset + Point<N, IndexT>::from(delta), args...);
    }

    output = cub::BlockReduce<OutputT, 256>().Sum(output);

    if (threadIdx.x == 0) {
        output_addr[blockIdx.x] = output;
    }
}
}  // namespace detail

/// Launcher (for use with `Device::access`/`DeviceGuard::parallel_for`) that applies `fun` to
/// every point of the N-dimensional index space `shape`.
template<typename F, size_t N, typename OutputT>
class ParallelReduce {
  public:
    using index_type = default_index_type;

    explicit ParallelReduce(Shape<N, index_type> shape, F fun, unsigned int block_size = 256) :
        m_shape(shape),
        m_fun(std::move(fun)),
        m_block_size(block_size) {}

    template<typename... Args>
    void operator()(g_stream_t stream, kmm::ViewMut<OutputT> output, Args&&... args) const {
        launch_recur(stream, m_offset, m_shape, output.data(), args...);
    }

    template<typename... Args>
    void operator()(g_stream_t stream, OutputT* output_addr, Args&&... args) const {
        launch_recur(stream, m_offset, m_shape, output_addr, args...);
    }

    size_t num_outputs() const {
        return num_outputs_impl(m_shape);
    }

  private:
    size_t num_outputs_impl(Shape<N, index_type> shape) const {
        if (shape.is_empty()) {
            return 0;
        }

        size_t num_outputs = 0;
        uint32_t num_threads = 1;

        for (size_t i = 0; i < N; i++) {
            auto chunk = index_type(IndexMapper<uint32_t, N>::max_volume / num_threads);

            while (is_less(chunk, shape[i])) {
                Shape<N> head = shape;
                head[i] = chunk;
                num_outputs += num_outputs_impl(head);

                shape[i] -= chunk;
            }

            num_threads *= uint32_t(shape[i]);
        }

        uint32_t grid_size = div_ceil(num_threads, m_block_size);
        return num_outputs + grid_size;
    }

    template<typename... Args>
    OutputT* launch_recur(
        g_stream_t stream,
        Point<N, index_type> offset,
        Shape<N, index_type> shape,
        OutputT* output_addr,
        const Args&... args
    ) const {
        if (shape.is_empty()) {
            return output_addr;
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
                output_addr = launch_recur(stream, offset, head, output_addr, args...);

                offset[i] += chunk;
                shape[i] -= chunk;
            }

            num_threads *= uint32_t(shape[i]);
        }

        uint32_t grid_size = div_ceil(num_threads, m_block_size);
        auto mapper = IndexMapper<uint32_t, N>(Shape<N, uint32_t>::from(shape));

        if (offset == Point<N>::zero()) {
            detail::parallel_reduce_kernel<F, N, index_type, OutputT, std::decay_t<Args>...>
                <<<grid_size, m_block_size, 0, stream>>>(mapper, m_fun, output_addr, args...);
        } else {
            detail::parallel_reduce_offset_kernel<F, N, index_type, OutputT, std::decay_t<Args>...>
                <<<grid_size, m_block_size, 0, stream>>>(
                    offset,
                    mapper,
                    m_fun,
                    output_addr,
                    args...
                );
        }

        return output_addr + grid_size;
    }

    F m_fun;
    Point<N, index_type> m_offset;
    Shape<N, index_type> m_shape;
    uint32_t m_block_size;
};

}  // namespace kmm
