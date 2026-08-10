#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kmm/api/array_base.hpp"
#include "kmm/api/launch_arg.hpp"
#include "kmm/core/layout.hpp"
#include "kmm/core/view.hpp"
#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/requisition.hpp"

namespace kmm {

template<typename T, typename DomainT, typename PolicyT = RowMajor>
class DomainArray: public ArrayBase {
    // Grants every `DomainArray` instantiation access to every other's private constructor.
    template<typename, typename, typename>
    friend class DomainArray;

    // Grants `Reduce<DomainArray<T, DomainT, PolicyT>>::finalize()` access to the private raw
    // constructor, so it can hand back a `DomainArray` without going through a public constructor
    // that would (re-)trigger `Runtime::begin_reduction`.
    friend class Reduce<DomainArray<T, DomainT, PolicyT>>;

  public:
    using self_type = DomainArray<T, DomainT, PolicyT>;
    using element_type = T;
    using domain_type = DomainT;
    using policy_type = PolicyT;
    using layout_type = Layout<DomainT, PolicyT>;
    static constexpr size_t rank = layout_type::rank;
    using mapping_type = typename layout_type::mapping_type;
    using index_type = typename layout_type::index_type;
    using ndindex_type = typename layout_type::ndindex_type;
    using shape_type = typename layout_type::shape_type;
    using range_type = typename layout_type::range_type;
    using bounds_type = typename layout_type::bounds_type;
    using stride_type = typename layout_type::stride_type;
    using ndstrides_type = typename layout_type::ndstrides_type;

    /// The `DomainArray` sharing this array's element type but over a different `Layout`,
    template<typename NewLayoutT>
    using rebind_layout = DomainArray<  //
        T,
        typename NewLayoutT::domain_type,
        typename NewLayoutT::policy_type>;

    using zero_origin_type = rebind_layout<typename layout_type::zero_origin_type>;
    using move_origin_type = rebind_layout<typename layout_type::move_origin_type>;
    using reverse_axes_type = rebind_layout<typename layout_type::reverse_axes_type>;

    template<size_t Axis>
    using drop_axis_type = rebind_layout<typename layout_type::template drop_axis_type<Axis>>;

    template<size_t Axis>
    using insert_axis_type = rebind_layout<typename layout_type::template insert_axis_type<Axis>>;

    template<size_t Axis, typename SliceT>
    using slice_axis_type =
        rebind_layout<typename layout_type::template slice_axis_type<Axis, SliceT>>;

    template<typename... Slices>
    using slice_type = rebind_layout<typename layout_type::template slice_type<Slices...>>;

    DomainArray() = default;

    DomainArray(layout_type layout) : m_layout(layout) {}

    DomainArray(domain_type domain, policy_type policy = {}) :
        DomainArray(make_layout(domain, policy).normalize_offset()) {}

    DomainArray(Runtime runtime, layout_type layout, std::optional<T> fill_value = std::nullopt) :
        ArrayBase(Buffer(
            runtime,
            BufferLayout::for_type<T>(static_cast<size_t>(layout.offset_span().size())),
            "array",
            fill_value ? FillValue::from<T>(*fill_value) : FillValue {}
        )),
        m_layout(layout.normalize_offset()) {}

    DomainArray(
        Runtime runtime,
        domain_type domain,
        policy_type policy = {},
        std::optional<T> fill_value = std::nullopt
    ) :
        DomainArray(runtime, make_layout(domain, policy), fill_value) {}

    template<typename OtherDomainT, typename OtherPolicyT>
    DomainArray(const DomainArray<T, OtherDomainT, OtherPolicyT>& that) :
        ArrayBase(that),
        m_layout(that.layout()) {}

    const layout_type& layout() const noexcept {
        return m_layout;
    }

    const domain_type& domain() const noexcept {
        return m_layout.domain();
    }

    const mapping_type& mapping() const noexcept {
        return m_layout.mapping();
    }

    shape_type shape() const noexcept {
        return m_layout.shape();
    }

    /// The valid index range along the given axis.
    range_type bounds(size_t axis) const noexcept {
        return m_layout.bounds(axis);
    }

    /// The bounds (begin/end per axis) covered by this array.
    bounds_type bounds() const noexcept {
        return m_layout.bounds();
    }

    /// The first valid index along the given axis.
    index_type origin(size_t axis) const noexcept {
        return m_layout.origin(axis);
    }

    /// The first valid index along each axis.
    ndindex_type origin() const noexcept {
        return m_layout.origin();
    }

    stride_type stride(size_t axis) const noexcept {
        return m_layout.stride(axis);
    }

    /// The stride along each axis.
    ndstrides_type strides() const noexcept {
        return m_layout.strides();
    }

    index_type extent(size_t axis) const noexcept {
        return m_layout.extent(axis);
    }

    /// The start of the valid range along the given axis.
    index_type begin(size_t axis) const noexcept {
        return m_layout.begin(axis);
    }

    /// The end (exclusive) of the valid range along the given axis.
    index_type end(size_t axis) const noexcept {
        return m_layout.end(axis);
    }

    /// The start of the valid range along each axis.
    ndindex_type begin() const noexcept {
        return m_layout.begin();
    }

    /// The end (exclusive) of the valid range along each axis.
    ndindex_type end() const noexcept {
        return m_layout.end();
    }

    /// The total number of elements in this array.
    index_type size() const noexcept {
        return m_layout.size();
    }

    /// Whether this array covers zero elements.
    bool is_empty() const noexcept {
        return m_layout.is_empty();
    }

    /// Returns this array rebased so its domain starts at the zero index.
    zero_origin_type zero_origin() const noexcept {
        return {buffer(), m_layout.zero_origin()};
    }

    /// Returns this array shifted so it originates at the given index, keeping the same shape.
    move_origin_type move_origin(ndindex_type new_origin) const noexcept {
        return {buffer(), m_layout.move_origin(new_origin)};
    }

    /// Returns this array restricted to the intersection of its bounds and the given bounds.
    move_origin_type restrict_bounds(bounds_type new_bounds) const noexcept {
        return {buffer(), m_layout.restrict_bounds(new_bounds)};
    }

    /// Returns this array restricted along one axis to the intersection with [start, stop).
    template<size_t Axis>
    move_origin_type restrict_axis(index_type start, index_type stop) const noexcept {
        return {buffer(), m_layout.template restrict_axis<Axis>(start, stop)};
    }

    /// Returns this array with the given axis dropped, fixed at the given index.
    template<size_t Axis>
    drop_axis_type<Axis> drop_axis(index_type index) const noexcept {
        return {buffer(), m_layout.template drop_axis<Axis>(index)};
    }

    /// Returns this array with a new broadcast axis of the given extent inserted at the given
    /// position.
    template<size_t Axis>
    insert_axis_type<Axis> insert_axis(index_type extent = static_cast<index_type>(1))
        const noexcept {
        return {buffer(), m_layout.template insert_axis<Axis>(extent)};
    }

    /// Returns this array with the order of all axes reversed.
    reverse_axes_type reverse_axes() const noexcept {
        return {buffer(), m_layout.reverse_axes()};
    }

    /// Returns this array with the given axis sliced according to the given slice token (e.g.
    /// `all`, a `Range`, `new_axis`).
    template<size_t Axis, typename SliceT>
    slice_axis_type<Axis, SliceT> slice_axis(const SliceT& slice) const noexcept {
        return {buffer(), m_layout.template slice_axis<Axis>(slice)};
    }

    /// Returns this array with the given axis narrowed to the range [start, end).
    template<size_t Axis>
    self_type slice_axis(index_type start, index_type end) const noexcept {
        return self_type(buffer(), m_layout.template slice_axis<Axis>(start, end));
    }

    /// Returns this array sliced across all axes at once, one slice token per axis.
    template<typename... Slices>
    slice_type<Slices...> slice(const Slices&... slices) const noexcept {
        return {buffer(), m_layout.slice(slices...)};
    }

    template<typename SliceT>
    slice_axis_type<0, SliceT> operator[](const SliceT& slice) const noexcept {
        return {buffer(), m_layout.template slice_axis<0>(slice)};
    }

  private:
    DomainArray(Buffer buffer, layout_type layout) noexcept :
        ArrayBase(std::move(buffer)),
        m_layout(layout) {}

    layout_type m_layout {};
};

template<typename T, size_t N = 1, typename PolicyT = RowMajor>
using Array = DomainArray<T, Shape<N>, PolicyT>;

template<typename T, size_t N = 1, typename PolicyT = RowMajor>
using SubArray = DomainArray<T, Bounds<N>, PolicyT>;

namespace detail {

/// Shared implementation of `LaunchArg` for `DomainArray<T, DomainT, PolicyT>`, parameterized on the access
/// mode granted to the resolved view: `AccessMode::Read` yields a `DomainView<const T, LayoutT>`,
/// `AccessMode::ReadWrite` a `DomainView<T, LayoutT>`.
template<typename T, typename DomainT, typename PolicyT, AccessMode Mode>
class LaunchArgArray {
  public:
    using view_element_type = std::conditional_t<Mode == AccessMode::Read, const T, T>;
    using resolve_type = DomainView<view_element_type, Layout<DomainT, PolicyT>>;

    explicit LaunchArgArray(const DomainArray<T, DomainT, PolicyT>& array) : m_array(array) {}

    void acquire(Runtime& runtime, Requisition& req) {
        m_index = req.add(m_array.buffer().id(), Mode);
    }

    resolve_type resolve(Runtime& runtime, Requisition& req) {
        auto accessor = req.accessor(runtime, m_index);
        auto* data = static_cast<typename resolve_type::pointer>(accessor.address);
        return resolve_type(data, m_array.layout());
    }

    void release(Runtime& runtime, Requisition& req) {}

  private:
    DomainArray<T, DomainT, PolicyT> m_array;
    size_t m_index = 0;
};

}  // namespace detail

/// Read-only access to a `DomainArray` (the default when passed to `Context::scope` unwrapped).
template<typename T, typename DomainT, typename PolicyT>
class LaunchArg<DomainArray<T, DomainT, PolicyT>>:
    public detail::LaunchArgArray<T, DomainT, PolicyT, AccessMode::Read> {
  public:
    using detail::LaunchArgArray<T, DomainT, PolicyT, AccessMode::Read>::LaunchArgArray;
};

/// Read-only access to a `DomainArray` explicitly wrapped in `read(...)`.
template<typename T, typename DomainT, typename PolicyT>
class LaunchArg<Read<DomainArray<T, DomainT, PolicyT>>>:
    public detail::LaunchArgArray<T, DomainT, PolicyT, AccessMode::Read> {
  public:
    explicit LaunchArg(Read<DomainArray<T, DomainT, PolicyT>> arg) :
        detail::LaunchArgArray<T, DomainT, PolicyT, AccessMode::Read>(arg.value) {}
};

/// Read-write access to a `DomainArray` wrapped in `write(...)`.
template<typename T, typename DomainT, typename PolicyT>
class LaunchArg<Write<DomainArray<T, DomainT, PolicyT>>>:
    public detail::LaunchArgArray<T, DomainT, PolicyT, AccessMode::ReadWrite> {
  public:
    explicit LaunchArg(Write<DomainArray<T, DomainT, PolicyT>> arg) :
        detail::LaunchArgArray<T, DomainT, PolicyT, AccessMode::ReadWrite>(arg.value) {}
};

}  // namespace kmm
