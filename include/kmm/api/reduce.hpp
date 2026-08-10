#pragma once

#include <exception>
#include <utility>

#include "kmm/api/array.hpp"

namespace kmm {

/// A `DomainArray` that has been put into reduction mode -- constructed as
/// `Reduce(array)` or `Reduce(array, op)` (`op` defaults to `ReductionOp::Sum`). Only
/// reduce-mode access is possible on it (there is no `LaunchArg` specialization for plain
/// read/write access) -- attempting to read or write it is caught at compile time, not by an
/// implicit auto-finalize. Call `finalize()` to fold the accumulated values back into a regular
/// `DomainArray`, or just let it go out of scope: reduction mode is scoped to this object's
/// lifetime, so the destructor finalizes automatically if you don't. Move-only: exactly one
/// `Reduce` owns "when does this reduction end" at a time.
template<typename T, typename DomainT, typename PolicyT>
class Reduce<DomainArray<T, DomainT, PolicyT>> {
  public:
    using element_type = T;
    using domain_type = DomainT;
    using policy_type = PolicyT;
    using layout_type = Layout<DomainT, PolicyT>;

    Reduce() = default;

    explicit Reduce(
        const DomainArray<T, DomainT, PolicyT>& array,
        ReductionOp op = ReductionOp::Sum
    ) :
        m_layout(array.layout()),
        m_op(op),
        m_buffer(array.buffer()) {
        if (m_buffer) {
            m_buffer.runtime().begin_reduction(m_buffer.id(), data_type_of<T>(), op);
        }
    }

    Reduce(const Reduce&) = delete;
    Reduce& operator=(const Reduce&) = delete;

    Reduce(Reduce&& that) noexcept :
        m_layout(that.m_layout),
        m_op(that.m_op),
        m_buffer(std::move(that.m_buffer)) {}

    Reduce& operator=(Reduce&& that) noexcept {
        if (this != &that) {
            finish();
            m_buffer = std::move(that.m_buffer);
            m_layout = that.m_layout;
            m_op = that.m_op;
        }

        return *this;
    }

    ~Reduce() {
        finish();
    }

    const layout_type& layout() const noexcept {
        return m_layout;
    }

    ReductionOp op() const noexcept {
        return m_op;
    }

    /// Finalizes the reduction.
    DomainArray<T, DomainT, PolicyT> finalize() {
        m_buffer.runtime().finalize_reduction(m_buffer.id());
        return DomainArray<T, DomainT, PolicyT>(std::move(m_buffer), m_layout);
    }

  private:
    void finish() noexcept {
        if (!m_buffer) {
            return;
        }

        auto buffer = std::move(m_buffer);
        bool unwinding = std::uncaught_exceptions() > 0;

        try {
            if (!unwinding) {
                buffer.runtime().finalize_reduction(buffer.id());
            } else {
                buffer.runtime().rollback_reduction(buffer.id());
            }
        } catch (...) {
            buffer.runtime().rollback_reduction(buffer.id());
        }
    }

    layout_type m_layout {};
    ReductionOp m_op = ReductionOp::Sum;
    Buffer m_buffer;
};

/// Deduction guide: CTAD only ever considers the primary `Reduce<T>` template (intentionally
/// left undefined -- see `launch_arg.hpp`), so this specialization must spell out its own guide
/// for `Reduce(array)` / `Reduce(array, op)` to deduce through it.
template<typename T, typename DomainT, typename PolicyT>
Reduce(const DomainArray<T, DomainT, PolicyT>&, ReductionOp = ReductionOp::Sum)
    -> Reduce<DomainArray<T, DomainT, PolicyT>>;

/// Reduce-mode access to a `Reduce`-wrapped array. There is no `LaunchArg` specialization for
/// plain `DomainArray<...>` reduce access -- a buffer must be put into reduction mode first (see
/// `Reduce`'s constructor), which is what makes reduce access to it representable at all. Holds
/// a reference rather than a copy since `Reduce` is move-only; safe because every use is confined
/// to `Context::scope`'s own call, which completes before any temporary argument is destroyed.
template<typename T, typename DomainT, typename PolicyT>
class LaunchArg<Reduce<DomainArray<T, DomainT, PolicyT>>> {
  public:
    using resolve_type = DomainView<T, Layout<DomainT, PolicyT>>;

    explicit LaunchArg(const Reduce<DomainArray<T, DomainT, PolicyT>>& array) : m_array(array) {}

    void acquire(Runtime& runtime, Requisition& req) {
        m_index = req.add_reduction(m_array.buffer().id());
    }

    // Assumes `req` has already been polled to `Poll::Ready` by `Context::scope` -- see the
    // matching comment on `detail::LaunchArgArray::resolve`.
    resolve_type resolve(Runtime& runtime, Requisition& req) {
        auto accessor = req.accessor(runtime, m_index);
        auto* data = static_cast<typename resolve_type::pointer>(accessor.address);
        return resolve_type(data, m_array.layout());
    }

    void release(Runtime& runtime, Requisition& req) {}

  private:
    const Reduce<DomainArray<T, DomainT, PolicyT>>& m_array;
    size_t m_index = 0;
};

}  // namespace kmm
