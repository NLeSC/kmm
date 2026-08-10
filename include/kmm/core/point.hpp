#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/vec.hpp"

namespace kmm {

/// \addtogroup geometry
/// @{

/// An N-dimensional coordinate, backed by a `Vec<T, N>`.
///
/// A `Point` identifies a location within an N-dimensional index space (e.g. the position
/// of an element in an N-dimensional array).
template<size_t N, typename T = default_index_type>
class Point: public Vec<T, N> {
  public:
    using storage_type = Vec<T, N>;

    /// Construct point from vector.
    KMM_HOST_DEVICE
    constexpr Point(const storage_type& storage) : storage_type(storage) {}

    /// Construct point (0, 0, 0, ...).
    KMM_HOST_DEVICE
    constexpr Point() : storage_type(fill<N>(static_cast<T>(0))) {}

    /// Constructs a point from N values.
    template<typename... Ts, typename = assert_arity_t<N, T, Ts...>>
    KMM_HOST_DEVICE Point(T first, Ts&&... args) : storage_type {first, args...} {}

    /// Converts from a point of a different dimensionality/type, throwing on overflow.
    template<size_t M, typename U>
    KMM_HOST_DEVICE constexpr Point(const Point<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Point::from(that);
    }

    /// Builds a point from a vector, padding any missing dimensions with zero.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Point from(const Vec<U, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? static_cast<T>(that[i]) : static_cast<T>(0);
        }

        return Point(result);
    }

    /// Creates a point with every coordinate set to one.
    KMM_HOST_DEVICE
    static constexpr Point one() {
        return make_index_sequence<N>::template fill<Point>(static_cast<T>(1));
    }

    /// Creates a point with every coordinate set to zero.
    KMM_HOST_DEVICE
    static constexpr Point zero() {
        return make_index_sequence<N>::template fill<Point>(static_cast<T>(0));
    }

    /// Checks whether this point fits losslessly into a Point<M, U>.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (is_less(i, M)) {
                result &= is_convertible<U>((*this)[i]);
            } else {
                result &= is_equal((*this)[i], static_cast<T>(0));
            }
        }

        return result;
    }

    /// Returns coordinate i, or default_value if the axis is out of range.
    KMM_HOST_DEVICE
    T get_or_default(size_t i, T default_value = T {}) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(is_less(i, N))) {
                return (*this)[i];
            }
        }

        return default_value;
    }
};

template<typename... Ts>
Point(Ts&&...) -> Point<sizeof...(Ts)>;

/// Constructs a Point from the given coordinate values.
template<typename... Ts>
KMM_HOST_DEVICE Point<sizeof...(Ts)> point(const Ts&... values) {
    return Point<sizeof...(Ts)> {Vec<default_index_type, sizeof...(Ts)> {values...}};
}

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Point<N + M, T> concat(const Point<N, T>& lhs, const Point<M, T>& rhs) {
    return Point<N + M, T> {
        concat(static_cast<const Vec<T, N>&>(lhs), static_cast<const Vec<T, M>&>(rhs))};
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator==(const Point<N, T>& lhs, const Point<M, U>& rhs) {
    bool result = true;

    for (size_t i = 0; is_less(i, N) || is_less(i, M); i++) {
        result &= is_equal(lhs.get_or_default(i), rhs.get_or_default(i));
    }

    return result;
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator!=(const Point<N, T>& lhs, const Point<M, U>& rhs) {
    return !(lhs == rhs);
}

/// @}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct fmt::formatter<kmm::Point<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::Point<N, T>>: std::hash<kmm::Vec<T, N>> {};
#endif