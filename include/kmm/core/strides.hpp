#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/const_value.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/shape.hpp"
#include "kmm/core/type_utils.hpp"
#include "kmm/core/vec.hpp"

namespace kmm {

namespace detail {

template<size_t Axis, typename... Rest>
struct pack_element;

template<size_t Axis, typename T, typename... Rest>
struct pack_element<Axis, T, Rest...>: pack_element<Axis - 1, Rest...> {};

template<typename T, typename... Rest>
struct pack_element<0, T, Rest...> {
    using type = T;
};

template<typename StrideT, typename Seq>
struct strides_storage_type;

template<typename StrideT>
struct strides_storage_type<StrideT, TypeSequence<>> {
    static constexpr size_t rank = 0;

    template<size_t Axis>
    using axis_stride_type = ConstValue<StrideT {0}>;

    KMM_HOST_DEVICE
    constexpr StrideT get_dynamic(size_t axis) const noexcept {
        return 0;
    }

    template<size_t Axis>
    KMM_HOST_DEVICE constexpr axis_stride_type<Axis> get_static(ConstIndex<Axis>) const noexcept {
        return {};
    }
};

template<typename StrideT, typename... StridesT>
struct strides_storage_type<StrideT, TypeSequence<StridesT...>> {
    using storage_type = StrideT;
    static constexpr size_t rank = sizeof...(StridesT);

    template<typename T>
    struct convert_stride_impl {
        KMM_HOST_DEVICE
        static storage_type pack(const T& value) {
            return static_cast<storage_type>(value);
        }

        KMM_HOST_DEVICE
        static T unpack(const storage_type& value) {
            return static_cast<T>(value);
        }
    };

    template<auto Value>
    struct convert_stride_impl<ConstValue<Value>> {
        KMM_HOST_DEVICE
        static storage_type pack(ConstValue<Value>) {
            return static_cast<storage_type>(Value);
        }

        KMM_HOST_DEVICE
        static ConstValue<Value> unpack(storage_type) {
            return {};
        }
    };

    template<size_t Axis>
    using axis_stride_type =
        typename pack_element<Axis, StridesT..., ConstValue<StrideT {0}>>::type;

    KMM_HOST_DEVICE
    constexpr strides_storage_type(StridesT... strides) noexcept :
        m_values {convert_stride_impl<StridesT>::pack(strides)...} {}

    KMM_HOST_DEVICE
    constexpr storage_type get_dynamic(size_t axis) const noexcept {
        return axis < rank ? m_values[axis] : static_cast<storage_type>(0);
    }

    template<size_t Axis>
    KMM_HOST_DEVICE constexpr axis_stride_type<Axis> get_static(ConstIndex<Axis>) const noexcept {
        return convert_stride_impl<axis_stride_type<Axis>>::unpack(m_values[Axis]);
    }

    storage_type m_values[rank];
};

template<typename StrideT, auto... Values>
struct strides_storage_type<StrideT, TypeSequence<ConstValue<Values>...>> {
    static constexpr size_t rank = sizeof...(Values);

    template<size_t Axis>
    using axis_stride_type =
        typename pack_element<Axis, ConstValue<Values>..., ConstValue<StrideT {0}>>::type;

    KMM_HOST_DEVICE
    constexpr strides_storage_type(ConstValue<Values>... strides) noexcept {}

    KMM_HOST_DEVICE
    constexpr StrideT get_dynamic(size_t axis) const noexcept {
        constexpr StrideT static_values[rank] = {static_cast<StrideT>(Values)...};
        return axis < rank ? static_values[axis] : static_cast<StrideT>(0);
    }

    template<size_t Axis>
    KMM_HOST_DEVICE constexpr axis_stride_type<Axis> get_static(ConstIndex<Axis>) const noexcept {
        return {};
    }
};

}  // namespace detail

template<typename... StridesT>
class Strides:
    private detail::strides_storage_type<default_stride_type, TypeSequence<StridesT...>> {
    using base_type = detail::strides_storage_type<default_stride_type, TypeSequence<StridesT...>>;

  public:
    using stride_type = default_stride_type;
    static constexpr size_t rank = sizeof...(StridesT);

    template<size_t Axis>
    using axis_stride_type = typename base_type::template axis_stride_type<Axis>;

    /// Default constructor.
    template<size_t R = rank, typename = enable_if_t<(R > 0)>>
    KMM_HOST_DEVICE constexpr Strides() : base_type(StridesT {}...) {}

    /// Construct from the given strides.
    KMM_HOST_DEVICE
    constexpr Strides(StridesT... strides) : base_type(strides...) {}

    /// Construct from another strides object. Throws exception on mismatched types.
    template<typename... OtherStridesT, typename = enable_if_t<sizeof...(OtherStridesT) == rank>>
    KMM_HOST_DEVICE constexpr Strides(const Strides<OtherStridesT...>& that) :
        Strides(that, make_index_sequence<rank>()) {}

    /// Returns the `i`-th stride converted to `stride_type`.
    KMM_HOST_DEVICE
    constexpr stride_type operator[](size_t axis) const noexcept {
        return base_type::get_dynamic(axis);
    }

    /// Returns the `i`-th stride as
    template<size_t Axis>
    KMM_HOST_DEVICE constexpr axis_stride_type<Axis> get(ConstIndex<Axis> index = {})
        const noexcept {
        return base_type::get_static(index);
    }

    /// Returns the product `point[0] * strides[0] + point[1] * strides[1] + ...`.
    template<typename IndexT = stride_type>
    KMM_HOST_DEVICE constexpr ptrdiff_t linearize_offset(const Vec<IndexT, rank>& point
    ) const noexcept {
        return linearize_offset_impl(point, make_index_sequence<rank>());
    }

    /// Construct from another strides object. Throws exception on mismatched types.
    template<typename... OtherStridesT>
    KMM_HOST_DEVICE bool is_equal(const Strides<OtherStridesT...>& that) const noexcept {
        static constexpr size_t N = sizeof...(StridesT) > sizeof...(OtherStridesT)
            ? sizeof...(StridesT)
            : sizeof...(OtherStridesT);
        return is_equal_impl(that, make_index_sequence<N>());
    }

    KMM_HOST_DEVICE Vec<stride_type, rank> to_vec() const noexcept {
        return to_vec_impl(make_index_sequence<rank>());
    }

  private:
    template<typename... OtherStridesT, size_t... Is>
    KMM_HOST_DEVICE constexpr Strides(const Strides<OtherStridesT...>& that, IndexSequence<Is...>) :
        base_type(checked_cast<StridesT>(that.get(ConstIndex<Is> {}))...) {
        // Only reached via the public converting constructor, which is constrained to
        // sizeof...(OtherStridesT) == rank, so `that` always has exactly `rank` axes here.
    }

    template<typename IndexT, size_t... Is>
    KMM_HOST_DEVICE constexpr ptrdiff_t
    linearize_offset_impl(const Vec<IndexT, rank>& point, IndexSequence<Is...>) const noexcept {
        return (
            static_cast<ptrdiff_t>(0) + ...
            + (static_cast<ptrdiff_t>(point[Is])
               * static_cast<ptrdiff_t>(base_type::get_static(ConstIndex<Is>())))
        );
    }

    template<typename... OtherStridesT, size_t... Is>
    KMM_HOST_DEVICE bool is_equal_impl(const Strides<OtherStridesT...>& that, IndexSequence<Is...>)
        const noexcept {
        return (::kmm::is_equal(this->get(ConstIndex<Is>()), that.get(ConstIndex<Is>())) && ...);
    }

    template<size_t... Is>
    KMM_HOST_DEVICE Vec<stride_type, rank> to_vec_impl(IndexSequence<Is...>) const noexcept {
        return {(*this)[Is]...};
    }
};

template<typename... StridesT, typename... OtherStridesT>
KMM_HOST_DEVICE bool operator==(
    const Strides<StridesT...>& lhs,
    const Strides<OtherStridesT...>& rhs
) {
    return lhs.is_equal(rhs);
}

template<typename... StridesT, typename... OtherStridesT>
KMM_HOST_DEVICE bool operator!=(
    const Strides<StridesT...>& lhs,
    const Strides<OtherStridesT...>& rhs
) {
    return !(lhs == rhs);
}

template<typename... StridesT>
Strides<StridesT...> make_strides(const StridesT&... strides) {
    return {strides...};
}

namespace detail {
template<typename StrideT, size_t N, typename Indices = make_index_sequence<N>>
struct strides_repeat_impl;

template<typename StrideT, size_t N, size_t... Is>
struct strides_repeat_impl<StrideT, N, IndexSequence<Is...>> {
    // identity_t is needed just to expand the index sequence.
    using type = Strides<identity_t<StrideT, Is>...>;
};
}  // namespace detail

/// Alias for `Strides<StrideT, StrideT, StrideT, ...>`, where the argument is repeated `N` times.
template<typename StrideT, size_t N>
using StridesN = typename detail::strides_repeat_impl<StrideT, N>::type;

/// Indicates whether the strides are ordered from from the trailing to the leading axis (RowMajor)
/// or from the leading to the trailing axis (ColMajor).
enum struct MemoryOrder { RowMajor, ColMajor };

/// @}

namespace detail {

template<typename IndexT, typename Axes>
struct calculate_product;

template<typename IndexT, size_t... Is>
struct calculate_product<IndexT, IndexSequence<Is...>> {
    using type = IndexT;

    template<size_t N>
    KMM_HOST_DEVICE static type apply(const Shape<N, IndexT>& shape) {
        return (static_cast<IndexT>(shape[Is]) * ... * static_cast<IndexT>(1));
    }
};

template<typename IndexT>
struct calculate_product<IndexT, IndexSequence<>> {
    using type = ConstValue<IndexT {1}>;

    template<size_t N>
    KMM_HOST_DEVICE static type apply(const Shape<N, IndexT>& shape) {
        return {};
    }
};

template<MemoryOrder Order, typename IndexT, size_t N, typename = make_index_sequence<N>>
struct ordered_strides_impl;

template<typename IndexT, size_t N, size_t... Is>
struct ordered_strides_impl<MemoryOrder::ColMajor, IndexT, N, IndexSequence<Is...>> {
    using type = Strides<typename calculate_product<IndexT, make_index_sequence<Is>>::type...>;

    KMM_HOST_DEVICE static type apply(Shape<N, IndexT> shape, IndexT alignment) {
        if (N > 0 && alignment > 1) {
            shape[0] = round_up_to_multiple(shape[0], alignment);
        }

        return type(calculate_product<IndexT, make_index_sequence<Is>>::apply(shape)...);
    }
};

template<typename IndexT, size_t N, size_t... Is>
struct ordered_strides_impl<MemoryOrder::RowMajor, IndexT, N, IndexSequence<Is...>> {
    using type =
        Strides<typename calculate_product<IndexT, range_index_sequence_t<Is + 1, N>>::type...>;

    KMM_HOST_DEVICE static type apply(Shape<N, IndexT> shape, IndexT alignment) {
        if (N > 0 && alignment > 1) {
            shape[N - 1] = round_up_to_multiple(shape[N - 1], alignment);
        }

        return type(calculate_product<IndexT, range_index_sequence_t<Is + 1, N>>::apply(shape)...);
    }
};
}  // namespace detail

/// \addtogroup layout
/// @{

template<MemoryOrder Order, typename IndexT, size_t N>
using make_strides_t = typename detail::ordered_strides_impl<Order, IndexT, N>::type;

/// Build a Strides object from a given shape and memory order (RowMajor or ColMajor). The
/// contiguous axis will always be `ConstValue<IndexT{1}>`.
///
/// For example, given a shape (10, 5, 3)
///  - RowMajor: strides are (IndexT(15), IndexT(3), ConstValue<IndexT{1}>)
///  - ColMajor: strides are (ConstValue<IndexT{1}>, IndexT(10), IndexT(50))
template<MemoryOrder Order = MemoryOrder::RowMajor, typename IndexT, size_t N>
KMM_HOST_DEVICE make_strides_t<Order, IndexT, N> make_strides_from_shape(
    const Shape<N, IndexT>& shape,
    IndexT alignment = 1
) {
    return detail::ordered_strides_impl<Order, IndexT, N>::apply(shape, alignment);
}

/// @}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

namespace kmm {
template<typename... StridesT>
std::ostream& operator<<(std::ostream& stream, const Strides<StridesT...>& s) {
    return stream << s.to_vec();
}
}  // namespace kmm

template<typename... StridesT>
struct fmt::formatter<kmm::Strides<StridesT...>>: fmt::ostream_formatter {};
#endif
