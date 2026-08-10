#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/type_utils.hpp"
#include "kmm/core/vec.hpp"

namespace kmm {

/// \addtogroup geometry
/// @{

/// The extent (size) of an N-dimensional domain along each axis, backed by a `Vec<T, N>`.
///
/// A `Shape` describes how many elements exist along each axis of a domain (e.g. the
/// dimensions of an array).
template<size_t N, typename T = default_index_type>
class Shape: public Vec<T, N> {
  public:
    using storage_type = Vec<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Shape(const storage_type& storage) : storage_type(storage) {}

    /// Create an empty shape `(0, 0, ...)`
    KMM_HOST_DEVICE
    constexpr Shape() : storage_type(::kmm::fill<N>(static_cast<T>(0))) {}

    constexpr Shape(const Shape&) = default;
    constexpr Shape(Shape&&) noexcept = default;
    Shape& operator=(const Shape&) = default;
    Shape& operator=(Shape&&) noexcept = default;

    /// Create a shape `(first, args...)`
    template<typename... Ts, typename = assert_arity_t<N, T, Ts...>>
    KMM_HOST_DEVICE Shape(T first, Ts&&... args) : storage_type {first, args...} {}

    /// Create a shape from another shape. Throws on overflow.
    template<size_t M, typename U>
    KMM_HOST_DEVICE constexpr Shape(const Shape<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Shape::from(that);
    }

    /// Create a shape from another shape. Does not throw on overflow.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Shape from(const Vec<U, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? static_cast<T>(that[i]) : static_cast<T>(1);
        }

        return Shape(result);
    }

    /// Create a shape `(value, value, value, ...)`.
    KMM_HOST_DEVICE
    static constexpr Shape fill(T value) {
        return Shape(::kmm::fill<N>(value));
    }

    /// Create a shape `(1, 1, 1, ...)`.
    KMM_HOST_DEVICE
    static constexpr Shape one() {
        return fill(static_cast<T>(1));
    }

    /// Create a shape `(0, 0, 0, ...)`.
    KMM_HOST_DEVICE
    static constexpr Shape zero() {
        return fill(static_cast<T>(0));
    }

    /// Returns coordinate i, or default_value if the axis is out of range.
    KMM_HOST_DEVICE
    T get_or_default(size_t i, T default_value = static_cast<T>(1)) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(is_less(i, N))) {
                return (*this)[i];
            }
        }

        return default_value;
    }

    /// Checks whether this shape can be converted to `Shape<M, U>`.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (i < M) {
                result &= is_convertible<U>((*this)[i]);
            } else {
                result &= is_equal((*this)[i], static_cast<T>(1));
            }
        }

        return result;
    }

    /// Check if this shape is empty.
    ///
    /// A shape is empty if each axis is less than or equal to zero.
    KMM_HOST_DEVICE
    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= !(static_cast<T>(0) < (*this)[i]);
        }

        return result;
    }

    /// Returns the product of the extents of this shape.
    ///
    /// If one of the axis is negative, the returned value is zeor.
    KMM_HOST_DEVICE
    T volume() const {
        T result = static_cast<T>(1);

        if constexpr (N >= 1) {
            result = (*this)[0];

            for (size_t i = 1; is_less(i, N); i++) {
                result *= (*this)[i];
            }
        }

        return is_empty() ? static_cast<T>(0) : result;
    }

    /// Check if a point falls within this shape.
    ///
    /// This means that for each axis `p[i] >= 0` and `p[i] < shape[i]`.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE bool contains(const Point<M, U>& p) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N) && is_less(i, M); i++) {
            result &= !is_less(p[i], static_cast<U>(0)) && is_less(p[i], (*this)[i]);
        }

        if constexpr (N < M) {
            for (size_t i = N; is_less(i, M); i++) {
                result &= is_equal(p[i], static_cast<U>(0));
            }
        }

        if constexpr (N > M) {
            for (size_t i = M; is_less(i, N); i++) {
                result &= is_less(static_cast<T>(0), (*this)[i]);
            }
        }

        return result;
    }
};

template<typename... Ts>
Shape(Ts&&...) -> Shape<sizeof...(Ts)>;

/// Constructs a Shape from the given per-axis extents.
template<typename... Ts>
KMM_HOST_DEVICE Shape<sizeof...(Ts)> shape(const Ts&... values) {
    return Shape<sizeof...(Ts)> {Vec<default_index_type, sizeof...(Ts)> {values...}};
}

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Shape<N + M, T> concat(const Shape<N, T>& lhs, const Shape<M, T>& rhs) {
    return Shape<N + M, T> {
        concat(static_cast<const Vec<T, N>&>(lhs), static_cast<const Vec<T, M>&>(rhs))};
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator==(const Shape<N, T>& lhs, const Shape<M, U>& rhs) {
    bool result = true;

    for (size_t i = 0; is_less(i, N) || is_less(i, M); i++) {
        result &= is_equal(lhs.get_or_default(i), rhs.get_or_default(i));
    }

    return result;
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator!=(const Shape<N, T>& lhs, const Shape<M, U>& rhs) {
    return !(lhs == rhs);
}

/// @}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct fmt::formatter<kmm::Shape<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::Shape<N, T>>: std::hash<kmm::Vec<T, N>> {};
#endif