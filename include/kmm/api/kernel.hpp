#pragma once

#include <cuda_runtime.h>
#include <utility>

#include "kmm/api/device_context.hpp"

namespace kmm {

/// Launcher (for use with `DeviceContext::scope`/`DeviceContext::launch`) that launches a
/// `__global__` kernel function `fun` with a fixed grid/block/shared-memory configuration.
/// `fun` is launched on `context`'s stream as
/// `fun<<<grid_dim, block_dim, shared_mem, context>>>(args...)`, where `args...` are the views
/// resolved by `scope`.
template<typename F>
class Kernel {
  public:
    Kernel(F fun, dim3 grid_dim, dim3 block_dim, unsigned int shared_mem = 0) :
        m_fun(fun),
        m_grid_dim(grid_dim),
        m_block_dim(block_dim),
        m_shared_mem(shared_mem) {}

    template<typename... Args>
    void operator()(CUstream context, Args&&... args) const {
        m_fun<<<m_grid_dim, m_block_dim, m_shared_mem, context>>>(std::forward<Args>(args)...);
    }

  private:
    F m_fun;
    dim3 m_grid_dim;
    dim3 m_block_dim;
    unsigned int m_shared_mem;
};

/// Deduction guide: infers `F` from the kernel function passed to the constructor.
template<typename F>
Kernel(F, dim3, dim3, unsigned int = 0) -> Kernel<F>;

}  // namespace kmm
