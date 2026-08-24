#pragma once

#include "kmm/core/bounds.hpp"
#include "kmm/core/const_value.hpp"
#include "kmm/core/fshape.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/range.hpp"
#include "kmm/core/shape.hpp"
#include "kmm/core/strides.hpp"
#include "kmm/core/type_utils.hpp"

namespace kmm {

struct all_t {
    explicit all_t() = default;
};
constexpr static all_t all = all_t();

struct new_axis_t {
    explicit new_axis_t() = default;
};
constexpr static new_axis_t new_axis = new_axis_t();

namespace detail {

template<typename PolicyT, typename DomainT>
struct policy_traits;

template<
    typename PolicyT,
    typename DomainT,
    typename MappingT = typename policy_traits<PolicyT, DomainT>::mapping_type>
struct mapping_traits;

template<typename PolicyT, typename DomainT, typename... StridesT>
struct mapping_traits<PolicyT, DomainT, Strides<StridesT...>> {
    static constexpr size_t rank = sizeof...(StridesT);
    using mapping_type = Strides<StridesT...>;
    using stride_type = default_stride_type;

    KMM_HOST_DEVICE
    static constexpr stride_type stride(const mapping_type& mapping, size_t axis) {
        return mapping[axis];
    }

    template<typename IndexT>
    KMM_HOST_DEVICE static constexpr ptrdiff_t linearize_offset(
        const mapping_type& mapping,
        const Vec<IndexT, rank>& index
    ) {
        return mapping.linearize_offset(index);
    }

    template<typename Seq>
    struct unpack_sequence_helper;

    template<size_t... Is>
    struct unpack_sequence_helper<IndexSequence<Is...>> {
        using type = Strides<typename mapping_type::template axis_stride_type<Is>...>;

        KMM_HOST_DEVICE
        static constexpr type apply(const mapping_type& mapping) {
            return type {mapping.get(ConstIndex<Is>())...};
        }
    };

    template<size_t Axis>
    using drop_axis_type = typename unpack_sequence_helper<drop_index_sequence<rank, Axis>>::type;

    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr drop_axis_type<Axis> drop_axis(const mapping_type& mapping) {
        static_assert(Axis < rank, "axis out of bounds");
        return unpack_sequence_helper<drop_index_sequence<rank, Axis>>::apply(mapping);
    }

    template<size_t... Is>
    using permute_axes_type = Strides<typename mapping_type::template axis_stride_type<Is>...>;

    template<size_t... Is>
    KMM_HOST_DEVICE static constexpr permute_axes_type<Is...>
    permute_axes(const mapping_type& mapping, IndexSequence<Is...>) {
        return permute_axes_type<Is...> {mapping.get(ConstIndex<Is>())...};
    }

    template<size_t Axis, typename Before, typename After>
    struct insert_axis_helper;

    // The inserted axis always gets a compile-time-static stride of 0: it isn't backed by any
    // real memory dimension, so every index along it must map to the same offset (broadcast).
    template<size_t Axis, size_t... Before, size_t... After>
    struct insert_axis_helper<Axis, IndexSequence<Before...>, IndexSequence<After...>> {
        using type = Strides<
            typename mapping_type::template axis_stride_type<Before>...,
            ConstValue<stride_type {0}>,
            typename mapping_type::template axis_stride_type<After>...>;

        KMM_HOST_DEVICE
        static constexpr type apply(const mapping_type& mapping) {
            return type {
                mapping.get(ConstIndex<Before>())...,
                ConstValue<stride_type {0}> {},
                mapping.get(ConstIndex<After>())...};
        }
    };

    template<size_t Axis>
    using insert_axis_type = typename insert_axis_helper<
        Axis,
        make_index_sequence<Axis>,
        range_index_sequence_t<Axis, rank>>::type;

    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr insert_axis_type<Axis> insert_axis(const mapping_type& mapping
    ) {
        static_assert(Axis <= rank, "axis out of bounds");
        return insert_axis_helper<
            Axis,
            make_index_sequence<Axis>,
            range_index_sequence_t<Axis, rank>>::apply(mapping);
    }
};

/// Customization point mapping a stride policy (e.g. `RowMajor`, `ColMajor`) and a domain type to
/// the mapping it produces. The default definition below just forwards to `PolicyT`'s own
/// `mapping_type` member alias and `apply` member function, so any policy following that
/// (intrusive) convention keeps working unchanged; specialize this trait instead when a policy
/// type can't (or shouldn't) define those members itself, e.g. a foreign type.
template<typename PolicyT, typename DomainT>
struct policy_traits {
    using mapping_type = typename PolicyT::template mapping_type<DomainT>;

    KMM_HOST_DEVICE
    static mapping_type apply(const PolicyT& policy, const DomainT& domain) {
        return policy.apply(domain);
    }
};

/// A `Strides<...>` used directly as a policy is its own mapping, independent of the domain: this
/// lets a `Layout` be constructed from an already-computed mapping (as if it were a policy)
/// instead of only from a policy that derives one from the domain.
template<typename... StridesT, typename DomainT>
struct policy_traits<Strides<StridesT...>, DomainT> {
    using mapping_type = Strides<StridesT...>;

    KMM_HOST_DEVICE
    static mapping_type apply(const Strides<StridesT...>& policy, const DomainT&) {
        return policy;
    }
};

template<typename LayoutT, size_t Axis, typename SliceT>
struct slice_axis_impl;

template<typename LayoutT, size_t Axis>
struct slice_axis_impl<LayoutT, Axis, all_t> {
    // How many axes this token writes into the result at this position; slice_multi_impl uses
    // this to compute where the next token's axis index lands (next_axis = Axis + produced_axes).
    static constexpr size_t produced_axes = 1;
    using type = LayoutT;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, all_t) {
        return layout;
    }
};

template<typename LayoutT, size_t Axis>
struct slice_axis_impl<LayoutT, Axis, int> {
    static constexpr size_t produced_axes = 0;
    using type = typename LayoutT::template drop_axis_type<Axis>;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, int index) {
        return layout.template drop_axis<Axis>(static_cast<typename LayoutT::index_type>(index));
    }
};

template<typename LayoutT, size_t Axis>
struct slice_axis_impl<LayoutT, Axis, long> {
    static constexpr size_t produced_axes = 0;
    using type = typename LayoutT::template drop_axis_type<Axis>;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, long index) {
        return layout.template drop_axis<Axis>(static_cast<typename LayoutT::index_type>(index));
    }
};

template<typename LayoutT, size_t Axis>
struct slice_axis_impl<LayoutT, Axis, size_t> {
    static constexpr size_t produced_axes = 0;
    using type = typename LayoutT::template drop_axis_type<Axis>;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, size_t index) {
        return layout.template drop_axis<Axis>(static_cast<typename LayoutT::index_type>(index));
    }
};

template<typename LayoutT, size_t Axis, typename IndexT>
struct slice_axis_impl<LayoutT, Axis, Range<IndexT>> {
    static constexpr size_t produced_axes = 1;
    using type = LayoutT;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, Range<IndexT> range) {
        using index_type = typename LayoutT::index_type;
        return layout.template slice_axis<Axis>(
            static_cast<index_type>(range.start),
            static_cast<index_type>(range.stop)
        );
    }
};

template<typename LayoutT, size_t Axis>
struct slice_axis_impl<LayoutT, Axis, new_axis_t> {
    static constexpr size_t produced_axes = 1;
    using type = typename LayoutT::template insert_axis_type<Axis>;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, new_axis_t) {
        return layout.template insert_axis<Axis>();
    }
};

template<typename LayoutT, size_t Axis, typename... Slices>
struct slice_multi_impl;

// Base case: every slice token has been consumed.
template<typename LayoutT, size_t Axis>
struct slice_multi_impl<LayoutT, Axis> {
    static_assert(Axis == LayoutT::rank, "invalid number of axis slices");
    using type = LayoutT;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout) {
        return layout;
    }
};

// General case: consume the head slice token (HeadSliceT), then recurse on the rest (TailSlicesT...).
template<typename LayoutT, size_t Axis, typename SliceT, typename... Rest>
struct slice_multi_impl<LayoutT, Axis, SliceT, Rest...> {
    // Bounds-checking against LayoutT::rank is left to slice_axis_impl::apply itself (via
    // drop_axis/slice_axis/insert_axis's own static_asserts), since whether an existing axis is
    // required at all depends on the token (e.g. new_axis_t needs none).
    static_assert(Axis <= LayoutT::rank, "axis out of bounds");

    using new_layout = typename slice_axis_impl<LayoutT, Axis, SliceT>::type;
    static constexpr size_t next_axis =
        Axis + slice_axis_impl<LayoutT, Axis, SliceT>::produced_axes;
    using type = typename slice_multi_impl<new_layout, next_axis, Rest...>::type;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, const SliceT& head, const Rest&... tail) {
        return slice_multi_impl<new_layout, next_axis, Rest...>::apply(
            slice_axis_impl<LayoutT, Axis, SliceT>::apply(layout, head),
            tail...
        );
    }
};

// Specialized for performance: `all_t` is a no-op on the layout, so just skip straight to the
// next axis without going through slice_axis_impl<..., all_t>::apply at all.
template<typename LayoutT, size_t Axis, typename... Rest>
struct slice_multi_impl<LayoutT, Axis, all_t, Rest...> {
    using type = typename slice_multi_impl<LayoutT, Axis + 1, Rest...>::type;

    KMM_HOST_DEVICE
    static type apply(const LayoutT& layout, all_t, const Rest&... tail) {
        return slice_multi_impl<LayoutT, Axis + 1, Rest...>::apply(layout, tail...);
    }
};

}  // namespace detail

template<typename DomainT>
using domain_index_type = typename detail::domain_traits<DomainT>::index_type;

template<typename DomainT>
static constexpr size_t domain_rank = detail::domain_traits<DomainT>::rank;

namespace detail {

// Calls make_strides<Order,StrideT>(extent_0, extent_1, ...), reading each axis's extent
// out of a domain_traits-typed domain rather than taking them as separate arguments.
template<MemoryOrder Order, typename StrideT, typename DomainT, size_t... Is>
KMM_HOST_DEVICE make_strides_t<Order, StrideT, sizeof...(Is)>
make_strides_for_domain(const DomainT& domain, StrideT alignment, IndexSequence<Is...>) {
    Shape<sizeof...(Is), StrideT> extents = {
        checked_cast<StrideT>(domain_traits<DomainT>::extent(domain, Is))...};
    return make_strides_from_shape<Order>(extents, alignment);
}

}  // namespace detail

/// \addtogroup layout
/// @{

template<MemoryOrder order, size_t alignment = 1, typename StrideT = void>
struct LayoutPolicy {
    template<typename DomainT>
    using mapping_type = make_strides_t<  //
        order,
        nonvoid_t<StrideT, domain_index_type<DomainT>>,
        domain_rank<DomainT>>;

    template<typename DomainT>
    KMM_HOST_DEVICE mapping_type<DomainT> apply(const DomainT& domain) const noexcept {
        using stride_type = nonvoid_t<StrideT, domain_index_type<DomainT>>;

        return detail::make_strides_for_domain<order, stride_type>(
            domain,
            checked_cast<stride_type>(alignment),
            make_index_sequence<domain_rank<DomainT>>()
        );
    }
};

/// A stride policy laying out a domain in column-major order (the first axis varies fastest
/// in memory).
///
/// The leading axis is padded up to a multiple of `alignment` elements.
template<size_t alignment, typename StrideT = void>
struct ColMajorPadded: LayoutPolicy<MemoryOrder::ColMajor, alignment, StrideT> {};

/// A stride policy laying out a domain in row-major order (the last axis varies fastest in
/// memory).
///
/// The trailing axis padded up to a multiple of `alignment` elements.
template<size_t alignment, typename StrideT = void>
struct RowMajorPadded: LayoutPolicy<MemoryOrder::RowMajor, alignment, StrideT> {};

/// A stride policy laying out a domain in either column-major order or row-major order, depending
/// on a provided runtime value.
///
/// The contiguous axis is padded up to a multiple of `alignment` elements.
template<size_t alignment, typename StrideT = void>
struct StridedPadded {
    MemoryOrder order;

    KMM_HOST_DEVICE
    StridedPadded(MemoryOrder order = MemoryOrder::RowMajor) : order(order) {}

    template<typename DomainT>
    using mapping_type = StridesN<  //
        domain_index_type<DomainT>,
        domain_rank<DomainT>>;

    template<typename DomainT>
    KMM_HOST_DEVICE mapping_type<DomainT> apply(const DomainT& domain) const noexcept {
        if (order == MemoryOrder::RowMajor) {
            return RowMajorPadded<alignment, StrideT> {}.apply(domain);
        } else {
            return ColMajorPadded<alignment, StrideT> {}.apply(domain);
        }
    }
};

/// Column-major stride policy (the first axis varies fastest in memory).
struct ColMajor: LayoutPolicy<MemoryOrder::ColMajor, 1> {};

/// Row-major stride policy (the last axis varies fastest in memory).
struct RowMajor: LayoutPolicy<MemoryOrder::RowMajor, 1> {};

/// Policy that is either column-major or row-major major, depending on a runtime value.
struct Strided: StridedPadded<1> {
    KMM_HOST_DEVICE
    Strided(MemoryOrder order = MemoryOrder::RowMajor) : StridedPadded<1>(order) {}
};

/// Combines a domain (Shape or Bounds) with a stride mapping to describe how n-dimensional
/// indices translate to linear storage offsets. `PolicyT` is a stride policy (e.g. `RowMajor`,
/// `ColMajor`) from which the actual mapping is derived via `detail::policy_traits`; a mapping
/// (e.g. `Strides<...>`) can also be passed directly as `PolicyT`, since it acts as its own
/// (domain-independent) policy. Defaults to `RowMajor`.
template<typename DomainT, typename PolicyT = RowMajor>
class Layout {
  public:
    using self_type = Layout<DomainT, PolicyT>;
    using domain_type = DomainT;
    using policy_type = PolicyT;

    using domain_traits = detail::domain_traits<domain_type>;
    static constexpr size_t rank = domain_traits::rank;
    using index_type = typename domain_traits::index_type;
    using ndindex_type = Point<rank, index_type>;
    using shape_type = Shape<rank, index_type>;
    using range_type = Range<index_type>;
    using bounds_type = Bounds<rank, index_type>;

    using mapping_traits = detail::mapping_traits<policy_type, domain_type>;
    using mapping_type = typename mapping_traits::mapping_type;
    static_assert(mapping_traits::rank == rank, "domain and mapping must have the same rank");
    using stride_type = typename mapping_traits::stride_type;
    using ndstrides_type = Vec<stride_type, rank>;

    using zero_origin_type = Layout<shape_type, mapping_type>;
    using move_origin_type = Layout<bounds_type, mapping_type>;

    template<size_t Axis>
    using drop_axis_type = Layout<  //
        typename domain_traits::template drop_axis_type<Axis>,
        typename mapping_traits::template drop_axis_type<Axis>>;

    template<size_t Axis>
    using insert_axis_type = Layout<  //
        typename domain_traits::template insert_axis_type<Axis>,
        typename mapping_traits::template insert_axis_type<Axis>>;

    template<size_t... Is>
    using permute_axes_type = Layout<  //
        typename domain_traits::template permute_axes_type<Is...>,
        typename mapping_traits::template permute_axes_type<Is...>>;

  private:
    template<typename Seq>
    struct permute_axes_seq_type_helper;

    template<size_t... Is>
    struct permute_axes_seq_type_helper<IndexSequence<Is...>> {
        using type = permute_axes_type<Is...>;
    };

  public:
    using reverse_axes_type =
        typename permute_axes_seq_type_helper<reverse_index_sequence<rank>>::type;

    template<size_t I, size_t J>
    using swap_axes_type =
        typename permute_axes_seq_type_helper<swap_index_sequence<rank, I, J>>::type;

    using transpose_type = swap_axes_type<0, 1>;

    template<size_t Axis, size_t Pos>
    using move_axis_to_position_type = typename permute_axes_seq_type_helper<
        move_axis_to_position_index_sequence<rank, Axis, Pos>>::type;

    template<size_t Axis>
    using move_axis_to_front_type = move_axis_to_position_type<Axis, 0>;

    template<size_t Axis>
    using move_axis_to_back_type = move_axis_to_position_type<Axis, rank - 1>;

    template<size_t Axis, typename SliceT>
    using slice_axis_type = typename detail::slice_axis_impl<self_type, Axis, SliceT>::type;

    template<typename... Slices>
    using slice_type = typename detail::slice_multi_impl<self_type, 0, Slices...>::type;

    /// Constructs an empty (default-initialized domain and mapping) layout.
    KMM_HOST_DEVICE
    constexpr Layout() = default;

    /// Constructs a layout from a domain, a mapping, and an explicit base offset.
    KMM_HOST_DEVICE
    constexpr Layout(domain_type domain, mapping_type mapping, ptrdiff_t base_offset) :
        m_domain(domain),
        m_mapping(mapping),
        m_base_offset(base_offset) {}

    /// Constructs a layout from a domain and a mapping, deriving the base offset from the origin.
    KMM_HOST_DEVICE
    constexpr Layout(domain_type domain, mapping_type mapping) :
        m_domain(domain),
        m_mapping(mapping) {
        m_base_offset = -offset_span().start;
    }

    /// Constructs a layout from a domain and a policy. The template argument is needed since
    /// otherwise this constructor would be ambiguous with `Layout(domain_type, mapping_type)`.
    template<typename P = policy_type>
    KMM_HOST_DEVICE constexpr Layout(domain_type domain, P policy = {}) :
        m_domain(domain),
        m_mapping(detail::policy_traits<policy_type, domain_type>::apply(policy, domain)) {
        m_base_offset = -offset_span().start;
    }

    /// Converting constructor from a layout over a compatible domain/mapping type.
    template<typename OtherDomainT, typename OtherPolicyT>
    KMM_HOST_DEVICE constexpr Layout(const Layout<OtherDomainT, OtherPolicyT>& that) :
        m_domain(that.domain()),
        m_mapping(that.mapping()),
        m_base_offset(that.base_offset()) {}

    /// Returns the domain (shape or bounds) of this layout.
    KMM_HOST_DEVICE
    constexpr const domain_type& domain() const noexcept {
        return m_domain;
    }

    /// Returns the stride mapping of this layout.
    KMM_HOST_DEVICE
    constexpr const mapping_type& mapping() const noexcept {
        return m_mapping;
    }

    /// Returns the valid index range along the given axis, or a single-element range if out of rank.
    KMM_HOST_DEVICE
    range_type bounds(size_t axis) const noexcept {
        return is_less(axis, rank) ? domain_traits::bounds(m_domain, axis) : range_type::one();
    }

    /// Returns the first valid index along the given axis.
    KMM_HOST_DEVICE
    index_type origin(size_t axis) const noexcept {
        return begin(axis);
    }

    /// Returns the first valid index along each axis.
    KMM_HOST_DEVICE
    ndindex_type origin() const noexcept {
        return begin();
    }

    /// Returns the number of valid indices along the given axis, or 1 if out of rank.
    KMM_HOST_DEVICE
    index_type extent(size_t axis) const noexcept {
        return is_less(axis, rank) ? domain_traits::extent(m_domain, axis)
                                   : static_cast<index_type>(1);
    }

    /// Returns the start of the valid range along the given axis.
    KMM_HOST_DEVICE
    index_type begin(size_t axis) const noexcept {
        return bounds(axis).start;
    }

    /// Returns the end (exclusive) of the valid range along the given axis.
    KMM_HOST_DEVICE
    index_type end(size_t axis) const noexcept {
        return bounds(axis).stop;
    }

    /// Returns the start of the valid range along each axis.
    KMM_HOST_DEVICE
    ndindex_type begin() const noexcept {
        ndindex_type result;

        for (size_t axis = 0; is_less(axis, rank); axis++) {
            result[axis] = this->begin(axis);
        }

        return result;
    }

    /// Returns the end (exclusive) of the valid range along each axis.
    KMM_HOST_DEVICE
    ndindex_type end() const noexcept {
        ndindex_type result;

        for (size_t axis = 0; is_less(axis, rank); axis++) {
            result[axis] = this->end(axis);
        }

        return result;
    }

    /// Returns the extent (size) along each axis.
    KMM_HOST_DEVICE
    shape_type shape() const noexcept {
        shape_type result;

        for (size_t axis = 0; is_less(axis, rank); axis++) {
            result[axis] = this->extent(axis);
        }

        return result;
    }

    /// Returns the bounds (begin/end per axis) covered by this layout.
    KMM_HOST_DEVICE
    bounds_type bounds() const noexcept {
        // from_bounds wants a Point, and Point's constructor from a Vec is explicit (no
        // implicit conversion), so ndindex_type's Vec values need converting explicitly here.
        return bounds_type::from_bounds(
            Point<rank, index_type>(this->begin()),
            Point<rank, index_type>(this->end())
        );
    }

    /// Returns the total number of elements covered by this layout.
    KMM_HOST_DEVICE
    index_type size() const noexcept {
        return bounds().volume();
    }

    /// Returns whether this layout covers zero elements.
    KMM_HOST_DEVICE
    bool is_empty() const noexcept {
        return bounds().is_empty();
    }

    /// Returns whether the given index falls within this layout's bounds.
    KMM_HOST_DEVICE
    bool contains(const ndindex_type& p) const noexcept {
        return bounds().contains(Point<rank, index_type>(p));
    }

    /// Returns the stride along the given axis.
    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return mapping_traits::stride(m_mapping, axis);
    }

    /// Returns the stride along each axis.
    KMM_HOST_DEVICE
    ndstrides_type strides() const noexcept {
        ndstrides_type result;

        for (size_t axis = 0; is_less(axis, rank); axis++) {
            result[axis] = this->stride(axis);
        }

        return result;
    }

    /// Returns the constant offset independent of any index, e.g. from a non-zero domain origin.
    KMM_HOST_DEVICE
    constexpr ptrdiff_t base_offset() const noexcept {
        return m_base_offset;
    }

    /// Computes the offset corresponding to an n-dimensional index, relative to `base_offset()`.
    KMM_HOST_DEVICE
    ptrdiff_t local_offset(const ndindex_type& index) const noexcept {
        return mapping_traits::linearize_offset(m_mapping, index);
    }

    /// Returns the total offset to add to the raw data pointer to reach `index`:
    /// `base_offset() + local_offset(index)`.
    KMM_HOST_DEVICE
    ptrdiff_t offset(const ndindex_type& index) const noexcept {
        return base_offset() + local_offset(index);
    }

    KMM_HOST_DEVICE
    Range<ptrdiff_t> offset_span() const noexcept {
        ptrdiff_t lo = m_base_offset;
        ptrdiff_t hi = m_base_offset;

        if (!is_empty()) {
            for (size_t i = 0; i < rank; i++) {
                auto s = static_cast<ptrdiff_t>(stride(i));
                auto [a, b] = bounds(i);
                lo += static_cast<ptrdiff_t>(s >= 0 ? a : b - 1) * s;
                hi += static_cast<ptrdiff_t>(s >= 0 ? b : a - 1) * s;
            }
        }

        return {lo, hi};
    }

    /// Returns this layout with `delta` added to its base offset (domain and mapping unchanged).
    KMM_HOST_DEVICE
    constexpr self_type shift_offset(ptrdiff_t delta) const noexcept {
        return self_type {m_domain, m_mapping, m_base_offset + delta};
    }

    /// Returns this layout shifted so `offset_range()` starts at zero (i.e., becomes `(0, N)`).
    KMM_HOST_DEVICE
    self_type normalize_offset() const noexcept {
        return shift_offset(-offset_span().start);
    }

    /// Returns whether the strides of this layout where produced by the given policy.
    template<typename OtherPolicyT = policy_type>
    KMM_HOST_DEVICE bool is_mapping_from_policy(const OtherPolicyT& policy = {}) const noexcept {
        return m_mapping
            == detail::policy_traits<OtherPolicyT, domain_type>::apply(policy, m_domain);
    }

    /// Returns whether the layout's strides are contiguous in the given memory order.
    KMM_HOST_DEVICE
    bool is_contiguous(MemoryOrder order = MemoryOrder::RowMajor) const noexcept {
        if (order == MemoryOrder::RowMajor) {
            return m_mapping == make_strides_from_shape<MemoryOrder::RowMajor>(shape());
        } else {
            return m_mapping == make_strides_from_shape<MemoryOrder::ColMajor>(shape());
        }
    }

    /// Returns a copy of this layout with the domain replaced (mapping and base offset kept).
    template<typename NewDomainT>
    KMM_HOST_DEVICE Layout<NewDomainT, mapping_type> with_domain(const NewDomainT& new_domain
    ) const noexcept {
        return Layout<NewDomainT, mapping_type> {new_domain, m_mapping, m_base_offset};
    }

    /// Returns a copy of this layout with the mapping replaced (domain and base offset kept).
    template<typename NewMappingT>
    KMM_HOST_DEVICE Layout<domain_type, NewMappingT> with_mapping(const NewMappingT& new_mapping
    ) const noexcept {
        return Layout<domain_type, NewMappingT> {m_domain, new_mapping, m_base_offset};
    }

    /// Returns this layout rebased so its domain starts at the zero index.
    KMM_HOST_DEVICE
    zero_origin_type zero_origin() const noexcept {
        return zero_origin_type {shape(), m_mapping, m_base_offset + local_offset(origin())};
    }

    /// Returns this layout shifted so it originates at the given index, keeping the same shape.
    KMM_HOST_DEVICE
    move_origin_type move_origin(ndindex_type new_origin) const noexcept {
        auto new_domain = bounds_type::from_offset_size(Point<rank, index_type>::zero(), shape());
        auto offset_diff = local_offset(new_origin) - local_offset(origin());

        return move_origin_type {new_domain, m_mapping, m_base_offset + offset_diff};
    }

    /// Returns this layout restricted to the intersection of its bounds and the given bounds.
    KMM_HOST_DEVICE
    move_origin_type restrict_bounds(bounds_type new_bounds) const noexcept {
        return with_domain(new_bounds.intersection(bounds()));
    }

    /// Returns this layout restricted along one axis to the intersection with [start, stop).
    template<size_t Axis>
    KMM_HOST_DEVICE move_origin_type
    restrict_axis(index_type start, index_type stop) const noexcept {
        static_assert(Axis < rank, "axis out of bounds");
        auto new_bounds = bounds();
        new_bounds[Axis] = range_type {start, stop}.intersection(new_bounds[Axis]);
        return with_domain(new_bounds);
    }

    /// Returns this layout sliced to the given bounds and rebased to a zero origin.
    KMM_HOST_DEVICE
    zero_origin_type slice_bounds(bounds_type new_bounds) const noexcept {
        return with_domain(new_bounds).zero_origin();
    }

    /// Returns this layout with the given axis dropped, fixed at the given index.
    template<size_t Axis>
    KMM_HOST_DEVICE drop_axis_type<Axis> drop_axis(index_type index) const noexcept {
        static_assert(Axis < rank, "axis out of bounds");
        KMM_ASSERT(is_less(index, extent(Axis)));

        return drop_axis_type<Axis> {
            domain_traits::template drop_axis<Axis>(domain()),
            mapping_traits::template drop_axis<Axis>(mapping()),
            m_base_offset + static_cast<ptrdiff_t>(stride(Axis)) * static_cast<ptrdiff_t>(index)};
    }

    /// Returns this layout with a new broadcast axis of the given extent inserted at the given position.
    template<size_t Axis>
    KMM_HOST_DEVICE insert_axis_type<Axis> insert_axis(
        index_type extent = static_cast<index_type>(1)
    ) const noexcept {
        static_assert(Axis <= rank, "axis out of bounds");

        return insert_axis_type<Axis> {
            domain_traits::template insert_axis<Axis>(domain(), extent),
            mapping_traits::template insert_axis<Axis>(mapping()),
            m_base_offset};
    }

    /// Returns this layout with its axes reordered according to the given permutation, e.g.
    /// `permute_axes<2, 0, 1>()` moves the current axis 2 to position 0, axis 0 to position 1,
    /// and axis 1 to position 2.
    template<size_t... Is>
    KMM_HOST_DEVICE permute_axes_type<Is...> permute_axes(IndexSequence<Is...> = {})
        const noexcept {
        static_assert(sizeof...(Is) == rank, "permutation must contain exactly `rank` axes");
        static_assert(is_permutation<IndexSequence<Is...>>, "must be a valid permutation of axes");

        return permute_axes_type<Is...> {
            domain_traits::template permute_axes<Is...>(domain(), IndexSequence<Is...> {}),
            mapping_traits::template permute_axes<Is...>(mapping(), IndexSequence<Is...> {}),
            m_base_offset};
    }

    /// Returns this layout with the order of all axes reversed.
    KMM_HOST_DEVICE reverse_axes_type reverse_axes() const noexcept {
        return permute_axes(reverse_index_sequence<rank>());
    }

    /// Returns this layout with axes `I` and `J` swapped.
    template<size_t I, size_t J>
    KMM_HOST_DEVICE swap_axes_type<I, J> swap_axes() const noexcept {
        static_assert(I < rank && J < rank, "axis out of bounds");
        return permute_axes(swap_index_sequence<rank, I, J>());
    }

    /// Returns this layout with axes 0 and 1 swapped. Only valid for a rank-2 layout; use
    /// `swap_axes` or `permute_axes` for other ranks.
    KMM_HOST_DEVICE transpose_type transpose() const noexcept {
        static_assert(rank == 2, "transpose() requires a rank-2 layout");
        return swap_axes<0, 1>();
    }

    /// Returns this layout with the given axis moved to the given position, preserving the
    /// relative order of the remaining axes.
    template<size_t Axis, size_t Pos>
    KMM_HOST_DEVICE move_axis_to_position_type<Axis, Pos> move_axis_to_position() const noexcept {
        static_assert(Axis < rank && Pos < rank, "axis out of bounds");
        return permute_axes(move_axis_to_position_index_sequence<rank, Axis, Pos>());
    }

    /// Returns this layout with the given axis moved to the front (position 0), preserving the
    /// relative order of the remaining axes.
    template<size_t Axis>
    KMM_HOST_DEVICE move_axis_to_front_type<Axis> move_axis_to_front() const noexcept {
        return move_axis_to_position<Axis, 0>();
    }

    /// Returns this layout with the given axis moved to the back (position `rank - 1`),
    /// preserving the relative order of the remaining axes.
    template<size_t Axis>
    KMM_HOST_DEVICE move_axis_to_back_type<Axis> move_axis_to_back() const noexcept {
        return move_axis_to_position<Axis, rank - 1>();
    }

    /// Returns this layout with the given axis sliced according to the given slice token (e.g. `all`, a `Range`, `new_axis`).
    template<size_t Axis, typename SliceT>
    slice_axis_type<Axis, SliceT> slice_axis(const SliceT& slice) const noexcept {
        return detail::slice_axis_impl<self_type, Axis, SliceT>::apply(*this, slice);
    }

    /// Returns this layout with the given axis narrowed to the range [start, end).
    template<size_t Axis>
    KMM_HOST_DEVICE self_type slice_axis(index_type start, index_type end) const noexcept {
        static_assert(Axis < rank, "axis out of bounds");

        return self_type {
            domain_traits::template slice_axis<Axis>(domain(), start, end),
            m_mapping,
            m_base_offset + static_cast<ptrdiff_t>(stride(Axis)) * static_cast<ptrdiff_t>(start)};
    }

    /// Returns this layout sliced across all axes at once, one slice token per axis.
    template<typename... Slices>
    slice_type<Slices...> slice(const Slices&... slices) const noexcept {
        return detail::slice_multi_impl<self_type, 0, Slices...>::apply(*this, slices...);
    }

  private:
    KMM_ATTRIBUTE_NO_UNIQUE_ADDRESS domain_type m_domain {};
    KMM_ATTRIBUTE_NO_UNIQUE_ADDRESS mapping_type m_mapping {};
    ptrdiff_t m_base_offset = 0;
};

/// @}

/// \addtogroup layout
/// @{

/// Constructs a Layout for the given domain using the given stride policy.
template<typename PolicyT = RowMajor, typename DomainT>
KMM_HOST_DEVICE Layout<DomainT, PolicyT> make_layout(DomainT domain, PolicyT policy = {}) {
    return Layout<DomainT, PolicyT> {domain, policy};
}

/// @}

}  // namespace kmm

// Layout has no base class to delegate to, so this prints/hashes it as the tuple of its
// three constituent parts, each of which is itself printable/hashable (domain: Shape/Bounds,
// mapping: Strides, base_offset: a plain integer).
#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {
template<typename DomainT, typename PolicyT>
std::ostream& operator<<(std::ostream& stream, const Layout<DomainT, PolicyT>& layout) {
    return stream << "Layout(domain=" << layout.bounds() << ", strides=" << layout.strides()
                  << ", base_offset=" << layout.base_offset() << ")";
}
}  // namespace kmm

template<typename DomainT, typename PolicyT>
struct fmt::formatter<kmm::Layout<DomainT, PolicyT>>: fmt::ostream_formatter {};
#endif
