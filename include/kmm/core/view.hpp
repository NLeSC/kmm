#pragma once

#include "kmm/core/layout.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/core/type_utils.hpp"

namespace kmm {

/// Accessor tag: no restriction -- the view's data may be dereferenced from both host and device
/// code. This is the default.
struct AnyAccessor {
    template<typename T>
    KMM_HOST_DEVICE T& dereference(T* input) const noexcept {
        return *input;
    }
};

/// Accessor tag: the view's data may only be dereferenced from device (GPU) code. Dereferencing
/// it from host code panics at runtime with a clear message.
struct DeviceAccessor {
    template<typename T>
    KMM_HOST_DEVICE T& dereference(T* input) const noexcept {
#if KMM_IS_DEVICE
        return *input;
#else
        KMM_PANIC("cannot access device data from host code");
#endif
    }
};

namespace detail {

// Tag for DomainView's raw (pointer, layout) constructor: pointer is already offset-adjusted, so
// it must NOT be shifted again by layout.base_offset(). A free type (not a per-DomainView
// nested type) so it's the same type across every DomainView<T, LayoutT> instantiation --
// with_layout() below constructs a DomainView<T, NewLayoutT> that differs from the enclosing
// DomainView<T, LayoutT>.
struct view_raw_ctor_tag {};

template<typename ViewT, size_t... Is>
class ViewAccessor {
  public:
    static constexpr size_t rank = ViewT::rank;
    using index_type = typename ViewT::index_type;
    using ndindex_type = typename ViewT::ndindex_type;

    KMM_HOST_DEVICE
    constexpr ViewAccessor(const ViewT* view, identity_t<index_type, Is>... p) : m_view(view) {
        ((m_point[Is] = p), ...);
    }

    KMM_HOST_DEVICE
    static decltype(auto) index(const ViewT* view, identity_t<index_type, Is>... p) noexcept {
        if constexpr (sizeof...(Is) == rank) {
            return (*view)(p...);
        } else {
            return ViewAccessor(view, p...);
        }
    }

    KMM_HOST_DEVICE
    decltype(auto) operator[](index_type i) const noexcept {
        static constexpr size_t Axis = sizeof...(Is);
        return ViewAccessor<ViewT, Is..., Axis>::index(m_view, m_point[Is]..., i);
    }

  private:
    const ViewT* m_view;
    index_type m_point[rank] = {};
};

template<typename DerivedT, size_t N>
class DomainViewBase {
  public:
    // The `typename = ...` constrains this to index types that convert directly to a scalar
    // `index_type` (e.g. `int`), so it does not compete with `DomainView`'s non-template
    // `operator[](const ndindex_type&)` overload for `Vec`/`Point` indices (which do not convert
    // to a scalar `index_type`). `Self` (defaulted to `DerivedT`) makes the default argument
    // depend on this function template's own parameters rather than solely on the enclosing
    // class template's `DerivedT`, deferring its instantiation to call time -- `DerivedT` is
    // still incomplete while `DomainViewBase` is being instantiated as its base class.
    template<
        typename IndexT,
        typename Self = DerivedT,
        typename = decltype(static_cast<typename Self::index_type>(declval<IndexT>()))>
    KMM_HOST_DEVICE decltype(auto) operator[](IndexT i) const noexcept {
        const auto* self = static_cast<const DerivedT*>(this);
        using index_type = typename DerivedT::index_type;
        return ViewAccessor<DerivedT> {self}[static_cast<index_type>(i)];
    }
};

// Rank 0 has no axis to chain-index into. Deleted (rather than simply absent) so that
// DomainView<T, LayoutT>'s `using base_type::operator[];` always has a member to name, regardless
// of rank; actually calling this is a compile error with a clear cause.
template<typename DerivedT>
class DomainViewBase<DerivedT, 0> {
  public:
    template<typename IndexT>
    void operator[](IndexT) const = delete;
};

}  // namespace detail

/// \addtogroup views
/// @{

/// A dense or strided view over a multi-dimensional array.
template<typename T, typename LayoutT, typename AccessorT = AnyAccessor>
class DomainView: public detail::DomainViewBase<DomainView<T, LayoutT, AccessorT>, LayoutT::rank> {
    using base_type = detail::DomainViewBase<DomainView<T, LayoutT, AccessorT>, LayoutT::rank>;

  public:
    using base_type::operator[];

    using self_type = DomainView<T, LayoutT, AccessorT>;
    using layout_type = LayoutT;
    using element_type = T;
    using pointer = T*;
    using reference = T&;
    using accessor_type = AccessorT;

    static constexpr size_t rank = layout_type::rank;
    using index_type = typename layout_type::index_type;
    using ndindex_type = typename layout_type::ndindex_type;
    using shape_type = typename layout_type::shape_type;
    using bounds_type = typename layout_type::bounds_type;
    using stride_type = typename layout_type::stride_type;
    using ndstrides_type = typename layout_type::ndstrides_type;

    template<typename NewLayoutT>
    using rebind_layout = DomainView<T, NewLayoutT, AccessorT>;

    template<size_t Axis>
    using drop_axis_type = rebind_layout<typename layout_type::template drop_axis_type<Axis>>;

    template<size_t Axis>
    using insert_axis_type = rebind_layout<typename layout_type::template insert_axis_type<Axis>>;

    using reverse_axes_type = rebind_layout<typename layout_type::reverse_axes_type>;

    template<size_t... Is>
    using permute_axes_type =
        rebind_layout<typename layout_type::template permute_axes_type<Is...>>;

    template<size_t I, size_t J>
    using swap_axes_type = rebind_layout<typename layout_type::template swap_axes_type<I, J>>;

    using transpose_type = rebind_layout<typename layout_type::transpose_type>;

    template<size_t Axis, size_t Pos>
    using move_axis_to_position_type =
        rebind_layout<typename layout_type::template move_axis_to_position_type<Axis, Pos>>;

    template<size_t Axis>
    using move_axis_to_front_type =
        rebind_layout<typename layout_type::template move_axis_to_front_type<Axis>>;

    template<size_t Axis>
    using move_axis_to_back_type =
        rebind_layout<typename layout_type::template move_axis_to_back_type<Axis>>;

    using zero_origin_type = rebind_layout<typename layout_type::zero_origin_type>;
    using move_origin_type = rebind_layout<typename layout_type::move_origin_type>;

    template<size_t Axis, typename SliceT>
    using slice_axis_type =
        rebind_layout<typename layout_type::template slice_axis_type<Axis, SliceT>>;

    template<typename... Slices>
    using slice_type = rebind_layout<typename layout_type::template slice_type<Slices...>>;

    /// Constructs an empty (null) view.
    KMM_HOST_DEVICE
    constexpr DomainView() = default;

    /// Constructs a view from a pointer already adjusted for the layout's base offset.
    KMM_HOST_DEVICE
    constexpr DomainView(
        detail::view_raw_ctor_tag,
        pointer data,
        layout_type layout,
        AccessorT accessor = {}
    ) :
        m_data(data),
        m_layout(layout),
        m_accessor(accessor) {}

    /// Constructs a view over the given data pointer and layout.
    KMM_HOST_DEVICE
    constexpr DomainView(pointer data, layout_type layout, AccessorT accessor = {}) :
        DomainView(detail::view_raw_ctor_tag {}, data + layout.base_offset(), layout, accessor) {}

    /// Converting constructor from a view over a compatible element/layout type with the same
    /// accessor tag (converting between AnyAccessor and DeviceAccessor is not allowed). Carries
    /// over the source view's accessor value rather than default-constructing a new one, so any
    /// accessor state survives the conversion.
    template<typename U, typename OtherLayoutT>
    KMM_HOST_DEVICE constexpr DomainView(const DomainView<U, OtherLayoutT, AccessorT>& that) :
        DomainView(that.data(), that.layout(), that.accessor()) {}

    /// Returns the underlying data pointer.
    KMM_HOST_DEVICE
    constexpr pointer data() const noexcept {
        return m_data - m_layout.base_offset();
    }

    /// Returns the underlying data pointer at the given index.
    KMM_HOST_DEVICE
    constexpr pointer data_at(ndindex_type index) const noexcept {
        return &m_data[m_layout.local_offset(index)];
    }

    /// Returns the layout describing this view's domain, strides, and base offset.
    KMM_HOST_DEVICE
    constexpr const layout_type& layout() const noexcept {
        return m_layout;
    }

    /// Returns the accessor used to dereference this view's data.
    KMM_HOST_DEVICE
    constexpr accessor_type accessor() const noexcept {
        return m_accessor;
    }

    /// Returns the extent (size) along each axis.
    KMM_HOST_DEVICE
    shape_type shape() const noexcept {
        return m_layout.shape();
    }

    /// Returns the bounds (begin/end per axis) covered by this view.
    KMM_HOST_DEVICE
    bounds_type bounds() const noexcept {
        return m_layout.bounds();
    }

    /// Returns the extent along the given axis.
    KMM_HOST_DEVICE
    index_type extent(size_t axis) const noexcept {
        return m_layout.extent(axis);
    }

    /// Returns the total number of elements covered by this view.
    KMM_HOST_DEVICE
    index_type size() const noexcept {
        return m_layout.size();
    }

    /// Returns whether this view covers zero elements.
    KMM_HOST_DEVICE
    bool is_empty() const noexcept {
        return m_layout.is_empty();
    }

    /// Returns whether the given index falls within this view's bounds.
    KMM_HOST_DEVICE
    bool contains(const ndindex_type& index) const noexcept {
        return m_layout.contains(index);
    }

    /// Returns the stride along the given axis.
    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return m_layout.stride(axis);
    }

    /// Returns the stride along each axis.
    KMM_HOST_DEVICE
    ndstrides_type strides() const noexcept {
        return m_layout.strides();
    }

    /// Returns whether the view's strides are contiguous in the given memory order.
    KMM_HOST_DEVICE
    bool is_contiguous(MemoryOrder order = MemoryOrder::RowMajor) const noexcept {
        return m_layout.is_contiguous(order);
    }

    /// Returns a reference to the element at the given index, asserting it is in bounds.
    KMM_HOST_DEVICE
    reference access(const ndindex_type& index) const noexcept {
        KMM_DEBUG_ASSERT(contains(index));
        return m_accessor.dereference(data_at(index));
    }

    KMM_HOST_DEVICE reference operator[](const ndindex_type& index) const noexcept {
        return access(index);
    }

    /// Returns a reference to the element at the given per-axis indices.
    template<typename... Indices, typename = assert_arity_t<rank, Indices...>>
    KMM_HOST_DEVICE reference operator()(Indices... indices) const noexcept {
        return access(ndindex_type {static_cast<index_type>(indices)...});
    }

    /// Returns this view rebased so its domain starts at the zero index.
    KMM_HOST_DEVICE
    zero_origin_type zero_origin() const noexcept {
        return with_layout(m_layout.zero_origin());
    }

    /// Returns this view shifted so it originates at the given index, keeping the same shape.
    KMM_HOST_DEVICE
    move_origin_type move_origin(ndindex_type new_origin) const noexcept {
        return with_layout(m_layout.move_origin(new_origin));
    }

    /// Returns this view restricted to the intersection of its bounds and the given bounds.
    KMM_HOST_DEVICE
    move_origin_type restrict_bounds(bounds_type new_bounds) const noexcept {
        return with_layout(m_layout.restrict_bounds(new_bounds));
    }

    /// Returns this view restricted along one axis to the intersection with [start, stop).
    template<size_t Axis>
    KMM_HOST_DEVICE move_origin_type
    restrict_axis(index_type start, index_type stop) const noexcept {
        return with_layout(m_layout.template restrict_axis<Axis>(start, stop));
    }

    /// Returns this view with the given axis dropped, fixed at the given index.
    template<size_t Axis>
    KMM_HOST_DEVICE drop_axis_type<Axis> drop_axis(index_type index) const noexcept {
        return with_layout(m_layout.template drop_axis<Axis>(index));
    }

    /// Returns this view with a new broadcast axis of the given extent inserted at the given position.
    template<size_t Axis>
    KMM_HOST_DEVICE insert_axis_type<Axis> insert_axis(
        index_type extent = static_cast<index_type>(1)
    ) const noexcept {
        return with_layout(m_layout.template insert_axis<Axis>(extent));
    }

    /// Returns this view with the order of all axes reversed.
    KMM_HOST_DEVICE reverse_axes_type reverse_axes() const noexcept {
        return with_layout(m_layout.reverse_axes());
    }

    /// Returns this view with its axes reordered according to the given permutation, e.g.
    /// `permute_axes<2, 0, 1>()` moves the current axis 2 to position 0, axis 0 to position 1,
    /// and axis 1 to position 2.
    template<size_t... Is>
    KMM_HOST_DEVICE permute_axes_type<Is...> permute_axes(IndexSequence<Is...> seq = {})
        const noexcept {
        return with_layout(m_layout.template permute_axes<Is...>(seq));
    }

    /// Returns this view with axes `I` and `J` swapped.
    template<size_t I, size_t J>
    KMM_HOST_DEVICE swap_axes_type<I, J> swap_axes() const noexcept {
        return with_layout(m_layout.template swap_axes<I, J>());
    }

    /// Returns this view with axes 0 and 1 swapped. Only valid for a rank-2 view; use
    /// `swap_axes` or `permute_axes` for other ranks.
    KMM_HOST_DEVICE transpose_type transpose() const noexcept {
        return with_layout(m_layout.transpose());
    }

    /// Returns this view with the given axis moved to the given position, preserving the
    /// relative order of the remaining axes.
    template<size_t Axis, size_t Pos>
    KMM_HOST_DEVICE move_axis_to_position_type<Axis, Pos> move_axis_to_position() const noexcept {
        return with_layout(m_layout.template move_axis_to_position<Axis, Pos>());
    }

    /// Returns this view with the given axis moved to the front (position 0), preserving the
    /// relative order of the remaining axes.
    template<size_t Axis>
    KMM_HOST_DEVICE move_axis_to_front_type<Axis> move_axis_to_front() const noexcept {
        return with_layout(m_layout.template move_axis_to_front<Axis>());
    }

    /// Returns this view with the given axis moved to the back (position `rank - 1`),
    /// preserving the relative order of the remaining axes.
    template<size_t Axis>
    KMM_HOST_DEVICE move_axis_to_back_type<Axis> move_axis_to_back() const noexcept {
        return with_layout(m_layout.template move_axis_to_back<Axis>());
    }

    /// Returns this view with the given axis sliced according to the given slice token (e.g. `all`, a `Range`, `new_axis`).
    template<size_t Axis, typename SliceT>
    slice_axis_type<Axis, SliceT> slice_axis(const SliceT& slice) const noexcept {
        return with_layout(m_layout.template slice_axis<Axis>(slice));
    }

    /// Returns this view with the given axis narrowed to the range [start, end).
    template<size_t Axis>
    KMM_HOST_DEVICE self_type slice_axis(index_type start, index_type end) const noexcept {
        return with_layout(m_layout.template slice_axis<Axis>(start, end));
    }

    /// Returns this view sliced across all axes at once, one slice token per axis.
    template<typename... Slices>
    slice_type<Slices...> slice(const Slices&... slices) const noexcept {
        return with_layout(m_layout.slice(slices...));
    }

  private:
    template<typename NewLayoutT>
    rebind_layout<NewLayoutT> with_layout(NewLayoutT&& new_layout) const noexcept {
        auto delta = new_layout.base_offset() - m_layout.base_offset();
        return {detail::view_raw_ctor_tag {}, m_data + delta, new_layout, m_accessor};
    }

    pointer m_data = nullptr;
    KMM_ATTRIBUTE_NO_UNIQUE_ADDRESS LayoutT m_layout {};
    KMM_ATTRIBUTE_NO_UNIQUE_ADDRESS AccessorT m_accessor {};
};

/// A read-only view over a Shape<N, IndexT> domain with the given policy, for the common case
/// where callers just want an N-dimensional dense/strided view whose domain starts at index 0.
/// Use `ViewMut` for a mutable view.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type,
    typename AccessorT = AnyAccessor>
using View = DomainView<const T, Layout<Shape<N, IndexT>, PolicyT>, AccessorT>;

/// A mutable view over a Shape<N, IndexT> domain with the given policy. See `View` for the
/// read-only counterpart.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type,
    typename AccessorT = AnyAccessor>
using ViewMut = DomainView<T, Layout<Shape<N, IndexT>, PolicyT>, AccessorT>;

/// A read-only view over a Bounds<N, IndexT> domain with the given policy -- a view over a
/// sub-region/window of a larger domain, whose origin need not start at index 0 (unlike `View`,
/// which always starts at index 0). Use `SubViewMut` for a mutable view.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type,
    typename AccessorT = AnyAccessor>
using SubView = DomainView<const T, Layout<Bounds<N, IndexT>, PolicyT>, AccessorT>;

/// A mutable view over a Bounds<N, IndexT> domain with the given policy. See `SubView` for the
/// read-only counterpart.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type,
    typename AccessorT = AnyAccessor>
using SubViewMut = DomainView<T, Layout<Bounds<N, IndexT>, PolicyT>, AccessorT>;

/// A read-only view over a Shape<N, IndexT> domain whose data may only be dereferenced from
/// device (GPU) code. Use `DeviceViewMut` for a mutable view, or `DeviceSubView` for the
/// arbitrary-origin (Bounds<N, IndexT>) counterpart.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type>
using DeviceView = View<T, N, PolicyT, IndexT, DeviceAccessor>;

/// A mutable view over a Shape<N, IndexT> domain whose data may only be dereferenced from device
/// (GPU) code.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type>
using DeviceViewMut = ViewMut<T, N, PolicyT, IndexT, DeviceAccessor>;

/// A read-only view over a Bounds<N, IndexT> domain whose data may only be dereferenced from
/// device (GPU) code. Use `DeviceSubViewMut` for a mutable view.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type>
using DeviceSubView = SubView<T, N, PolicyT, IndexT, DeviceAccessor>;

/// A mutable view over a Bounds<N, IndexT> domain whose data may only be dereferenced from device
/// (GPU) code.
template<
    typename T,
    size_t N = 1,
    typename PolicyT = RowMajor,
    typename IndexT = default_index_type>
using DeviceSubViewMut = SubViewMut<T, N, PolicyT, IndexT, DeviceAccessor>;

/// Constructs a mutable ViewMut over the given data pointer and shape. Whether the result is
/// read-only follows T's own constness (e.g. passing a `const int*` yields a read-only view),
/// exactly like ViewMut<T, ...> does.
template<
    typename PolicyT = RowMajor,
    typename AccessorT = AnyAccessor,
    typename T,
    size_t N,
    typename IndexT = default_index_type>
KMM_HOST_DEVICE ViewMut<T, N, PolicyT, IndexT, AccessorT> make_view(
    T* data,
    Shape<N, IndexT> shape,
    PolicyT policy = {}
) {
    return {data, make_layout(shape, policy)};
}

/// @}

}  // namespace kmm

// DomainView has no base class to delegate to (other than the internal ViewAccessor/DomainViewBase
// storage helpers), so this prints/hashes it as the tuple of its two constituent parts (the raw
// pointer and the layout), matching how Layout itself is printed.
#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

namespace kmm {
template<typename T, typename LayoutT, typename AccessorT>
std::ostream& operator<<(std::ostream& stream, const DomainView<T, LayoutT, AccessorT>& view) {
    return stream << "DomainView(data=" << static_cast<const void*>(view.data())
                  << ", layout=" << view.layout() << ")";
}
}  // namespace kmm

template<typename T, typename LayoutT, typename AccessorT>
struct fmt::formatter<kmm::DomainView<T, LayoutT, AccessorT>>: fmt::ostream_formatter {};
#endif
