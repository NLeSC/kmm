#pragma once

#include <type_traits>
#include <utility>

#include "kmm/api/device.hpp"
#include "kmm/api/kernel.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/shape.hpp"

namespace kmm {

namespace detail {
/// Unravels the calling thread's linear index into a `Point<N>` within `shape` (row-major: the
/// last axis varies fastest) and, if that point falls within `shape`, calls `fun(point,
/// args...)`.
template<typename F, size_t N, typename IndexT, typename... Args>
__global__ void parallel_for_kernel(Shape<N, IndexT> shape, F fun, Args... args) {
    IndexT linear_index = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (linear_index >= shape.volume()) {
        return;
    }

    Point<N, IndexT> point;
    for (size_t i = N; i-- > 0;) {
        point[i] = linear_index % shape[i];
        linear_index /= shape[i];
    }

    fun(point, args...);
}
}  // namespace detail

/// Launcher (for use with `Device::access`/`DeviceGuard::parallel_for`) that applies `fun` to
/// every point of the N-dimensional index space `shape`, one GPU thread per point. `fun` is
/// called as `fun(point, args...)`, where `point` is the thread's `Point<N>` coordinate and
/// `args...` are the views resolved by `scope`.
template<typename F, size_t N>
class ParallelFor {
  public:
    using index_type = int;

    explicit ParallelFor(Shape<N> shape, F fun, unsigned int block_size = 256) :
        m_shape(shape),
        m_fun(std::move(fun)),
        m_block_size(block_size) {}

    template<typename... Args>
    void operator()(GPUStream context, Args&&... args) const {
        auto n = checked_cast<int>(m_shape.volume());

        if (n == 0) {
            return;
        }

        int grid_size = (n + m_block_size - 1) / m_block_size;

        Kernel<decltype(&detail::parallel_for_kernel<F, N, index_type, std::decay_t<Args>...>)>
            kernel(
                &detail::parallel_for_kernel<F, N, index_type, std::decay_t<Args>...>,
                dim3(grid_size),
                dim3(m_block_size)
            );

        kernel(context, m_shape, m_fun, std::forward<Args>(args)...);
    }

  private:
    Shape<N> m_shape;
    F m_fun;
    unsigned int m_block_size;
};

/// Deduction guide: infers `F` and `N` from the shape/functor passed to the constructor.
template<typename F, size_t N>
ParallelFor(Shape<N>, F, unsigned int = 256) -> ParallelFor<F, N>;

}  // namespace kmm
