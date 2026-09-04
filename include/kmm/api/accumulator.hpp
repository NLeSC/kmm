#pragma once

#include <utility>

#include "kmm/api/array.hpp"

namespace kmm {

/// A distributed reduction target: freshly allocated storage that kernels on one or more devices
/// add their contributions into. Each device writes its own partial result; the runtime combines
/// the per-device partials into the final array when `finalize()` is called.
template<typename T, typename LayoutT>
class NDAccumulator {
    KMM_NOT_COPYABLE(NDAccumulator)

  public:
    using element_type = T;
    using layout_type = LayoutT;
    using domain_type = typename layout_type::domain_type;
    using policy_type = typename layout_type::policy_type;

    /// Wraps a freshly allocated (uninitialized) `array` as a reduction target combining writes
    /// using `op`. `array` must not have been written to yet.
    explicit NDAccumulator(NDArray<T, LayoutT> array, ReductionOp op = ReductionOp::Sum) :
        m_array(std::move(array)),
        m_op(op) {
        m_array.buffer().runtime().begin_reduction(m_array.buffer().id(), data_type_of<T>(), op);
    }

    ~NDAccumulator() {
        if (m_array.buffer()) {
            m_array.buffer().runtime().rollback_reduction(m_array.buffer().id());
        }
    }

    const layout_type& layout() const noexcept {
        return m_array.layout();
    }

    ReductionOp op() const noexcept {
        return m_op;
    }

    const Buffer& buffer() const noexcept {
        return m_array.buffer();
    }

    /// Combines everything written to this accumulator so far and returns the result.
    /// Must be called at most once; after this, the `NDAccumulator` should be discarded.
    NDArray<T, LayoutT> finalize() {
        KMM_ASSERT(m_array.buffer());
        m_array.buffer().runtime().finalize_reduction(m_array.buffer().id());
        return std::exchange(m_array, NDArray<T, LayoutT>());
    }

  private:
    NDArray<T, LayoutT> m_array;
    ReductionOp m_op;
};

template<typename T, typename LayoutT>
class LaunchArg<NDAccumulator<T, LayoutT>> {
  public:
    using resolve_type = NDView<T, LayoutT>;

    explicit LaunchArg(const NDAccumulator<T, LayoutT>& array) : m_array(array) {}

    void acquire(Runtime& runtime, ResourceRequest& requests, MemoryId memory_id) {
        m_index = requests.add(memory_id, m_array.buffer().id(), AccessMode::Reduce);
    }

    resolve_type resolve(Runtime& runtime, const ResourceGrant& grant) {
        auto accessor = grant.accessor(m_index);
        auto* data = static_cast<typename resolve_type::pointer>(accessor.address);
        return resolve_type(data, m_array.layout());
    }

    void release(Runtime& runtime) {}

  private:
    const NDAccumulator<T, LayoutT>& m_array;
    size_t m_index = 0;
};

template<typename T, size_t N = 1, typename PolicyT = RowMajor>
using Accumulator = NDAccumulator<T, Layout<Shape<N>, PolicyT>>;

// `Reduce` / `ReduceInto` / `reduce(...)` -- the launch-argument side of reductions -- live in
// "kmm/api/reduce.hpp".

}  // namespace kmm
