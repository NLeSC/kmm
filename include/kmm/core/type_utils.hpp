#pragma once

#include "kmm/core/const_value.hpp"
#include "kmm/core/macros.hpp"

namespace kmm {

using size_t = decltype(sizeof(int));
using ptrdiff_t = decltype(static_cast<int*>(nullptr) - static_cast<int*>(nullptr));
using default_index_type = signed long long;  // int64_t
using default_stride_type = ptrdiff_t;  //

namespace detail {
template<bool Cond, typename TrueT, typename FalseT>
struct conditional_type_impl {
    using type = TrueT;
};

template<typename TrueT, typename FalseT>
struct conditional_type_impl<false, TrueT, FalseT> {
    using type = FalseT;
};
}  // namespace detail

template<bool Cond, typename TrueT, typename FalseT>
using conditional_t = typename detail::conditional_type_impl<Cond, TrueT, FalseT>::type;

namespace detail {
template<bool Cond, typename T = void>
struct enable_if_type_impl {};

template<typename T>
struct enable_if_type_impl<true, T> {
    using type = T;
};
}  // namespace detail

template<bool Cond, typename T = void>
using enable_if_t = typename detail::enable_if_type_impl<Cond, T>::type;

template<size_t N, typename... ArgsT>
using assert_arity_t = typename detail::enable_if_type_impl<N == sizeof...(ArgsT)>::type;

template<typename T, auto...>
using identity_t = T;

namespace detail {
// Usable only in unevaluated contexts (e.g. `decltype`); never defined/called.
template<typename T>
T&& declval() noexcept;
}  // namespace detail

template<auto...>
using void_t = void;

namespace detail {
template<typename T, typename... Rest>
struct nonvoid_impl {
    using type = T;
};

template<typename... Rest>
struct nonvoid_impl<void, Rest...>: nonvoid_impl<Rest...> {};
}  // namespace detail

/// Resolves to the first non-`void` type in `Ts...`.
template<typename... Ts>
using nonvoid_t = typename detail::nonvoid_impl<Ts...>::type;

template<typename... Ts>
struct TypeSequence {};

template<size_t... Indices>
struct IndexSequence {
    KMM_HOST_DEVICE
    static constexpr size_t size() noexcept {
        return sizeof...(Indices);
    }

    /// Returns `fun(0) && fun(1) && fun(2) && ...`
    template<typename F>
    KMM_HOST_DEVICE static bool all(F fun) {
        return ((fun(ConstIndex<Indices>())) && ...);
    }

    /// Calls `fun(0); fun(1); fun(2); ...`
    template<typename T, typename F>
    KMM_HOST_DEVICE static void for_each(F fun) {
        ((fun(ConstIndex<Indices>())), ...);
    }

    /// Constructs `T` from `T{fun(0), fun(1), fun(2), ...}`.
    template<typename T, typename F>
    KMM_HOST_DEVICE static T construct(F fun) {
        return {fun(ConstIndex<Indices>())...};
    }

    /// Constructs `T` from `T{value, value, value, ...}`.
    template<typename T, typename V>
    KMM_HOST_DEVICE static T fill(const V& value) {
        return {(ConstIndex<Indices>(), value)...};
    }
};

namespace detail {
template<size_t Begin, size_t End, size_t... Indices>
struct make_index_sequence_helper: make_index_sequence_helper<Begin + 1, End, Indices..., Begin> {};

template<size_t N, size_t... Indices>
struct make_index_sequence_helper<N, N, Indices...> {
    using type = IndexSequence<Indices...>;
};

// Shifts every index in `Seq` up by `Offset` (e.g. Offset=2, IndexSequence<0,1> -> IndexSequence<2,3>).
template<size_t Offset, typename Seq>
struct offset_index_sequence;

template<size_t Offset, size_t... Is>
struct offset_index_sequence<Offset, IndexSequence<Is...>> {
    using type = IndexSequence<(Is + Offset)...>;
};

template<typename Src, typename Dst = IndexSequence<>>
struct reverse_index_sequence_helper {
    using type = Dst;
};

template<size_t First, size_t... Rest, size_t... Indices>
struct reverse_index_sequence_helper<IndexSequence<First, Rest...>, IndexSequence<Indices...>>:
    reverse_index_sequence_helper<IndexSequence<Rest...>, IndexSequence<First, Indices...>> {};

template<size_t Axis, typename Seq>
struct drop_index_sequence_helper;

template<size_t Axis, size_t... Indices>
struct drop_index_sequence_helper<Axis, IndexSequence<0, Indices...>> {
    using type = IndexSequence<(Indices < Axis + 1 ? Indices - 1 : Indices)...>;
};

template<size_t I, size_t J, typename Seq>
struct swap_index_sequence_helper;

template<size_t I, size_t J, size_t... Indices>
struct swap_index_sequence_helper<I, J, IndexSequence<Indices...>> {
    using type = IndexSequence<(Indices == I ? J : (Indices == J ? I : Indices))...>;
};

constexpr size_t move_axis_to_position_index(size_t p, size_t axis, size_t pos) {
    // At position p, we get the new axis.
    // All other axes either shift right (if p in axis...pos) or shift left (p in pos...axis).
    return p == pos ? axis : p + size_t(axis <= p && p <= pos) - size_t(pos <= p && p <= axis);
}

template<size_t Axis, size_t Pos, typename Seq>
struct move_axis_to_position_index_sequence_helper;

template<size_t Axis, size_t Pos, size_t... Ps>
struct move_axis_to_position_index_sequence_helper<Axis, Pos, IndexSequence<Ps...>> {
    using type = IndexSequence<move_axis_to_position_index(Ps, Axis, Pos)...>;
};
}  // namespace detail

/// Alias for IndexSequence<0, 1, ..., N-1>
template<size_t N>
using make_index_sequence = typename detail::make_index_sequence_helper<0, N>::type;

/// Alias for IndexSequence<Begin, Begin + 1, ..., End -2, End - 1>
template<size_t Begin, size_t End>
using range_index_sequence_t = typename detail::make_index_sequence_helper<Begin, End>::type;

/// Alias for IndexSequence<N-1, N-2, ..., 0>
template<size_t N>
using reverse_index_sequence =
    typename detail::reverse_index_sequence_helper<make_index_sequence<N>>::type;

/// Alias for IndexSequence<0, 1, ..., Axis-1, Axis+1, ..., N-1>
template<size_t N, size_t Axis>
using drop_index_sequence =
    typename detail::drop_index_sequence_helper<Axis, make_index_sequence<N>>::type;

/// Alias for IndexSequence<0, 1, ..., N-1> with the values at positions `I` and `J` swapped.
template<size_t N, size_t I, size_t J>
using swap_index_sequence =
    typename detail::swap_index_sequence_helper<I, J, make_index_sequence<N>>::type;

/// Alias for IndexSequence<0, 1, ..., N-1> with axis `Axis` moved to position `Pos`, preserving
/// the relative order of the other axes.
template<size_t N, size_t Axis, size_t Pos>
using move_axis_to_position_index_sequence = typename detail::
    move_axis_to_position_index_sequence_helper<Axis, Pos, make_index_sequence<N>>::type;

namespace detail {
template<typename Seq, size_t N>
struct is_partial_permutation_impl {};

template<size_t N>
struct is_partial_permutation_impl<IndexSequence<>, N> {
    static constexpr bool value = true;
};

template<size_t First, size_t... Rest, size_t N>
struct is_partial_permutation_impl<IndexSequence<First, Rest...>, N> {
    static constexpr bool value = ((First != Rest) && ...) && (First < N)
        && is_partial_permutation_impl<IndexSequence<Rest...>, N>::value;
};
}  // namespace detail

/// True if `Seq` is an `IndexSequence` representing partial permutation for N. This means it
/// contains no duplicate indices and each index is less than N.
template<typename Seq, size_t N>
constexpr bool is_partial_permutation = detail::is_partial_permutation_impl<Seq, N>::value;

/// True if `Seq` is an `IndexSequence` representing a permutation of length N. This means it
/// contains N indices, no duplicates, and each index is less than N. In other words, it contains
/// the indices `0, 1, ..., N-1` in any arbitrary order.
template<typename Seq>
constexpr bool is_permutation = is_partial_permutation<Seq, Seq::size()>;

}  // namespace kmm