#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/core/type_utils.hpp"

namespace kmm {

/// \addtogroup geometry
/// @{

/// A fixed-size array of `N` values of type `T`.
///
/// This is the plain-storage building block used by `Point`, `Shape`, and `Bounds`: those
/// types inherit from `Vec` to add domain-specific semantics (coordinates, extents, ranges)
/// on top of simple indexed element access. Specializations exist for `N` in `{0, 1, 2, 3, 4}`
/// so that `x`/`y`/`z`/`w` accessors are available for low-dimensional vectors.
template<typename T, size_t N>
struct Vec {
    KMM_HOST_DEVICE
    constexpr T& operator[](size_t i) {
        KMM_ASSERT(i < N);
        return values[i];
    }

    KMM_HOST_DEVICE
    constexpr const T& operator[](size_t i) const {
        KMM_ASSERT(i < N);
        return values[i];
    }

    T values[N];
};

template<typename T>
struct Vec<T, 0> {
    KMM_HOST_DEVICE
    constexpr T& operator[](size_t i) {
        KMM_PANIC("index out of bounds");
    }

    KMM_HOST_DEVICE
    constexpr const T& operator[](size_t i) const {
        KMM_PANIC("index out of bounds");
    }
};

template<typename T>
struct Vec<T, 1> {
    KMM_HOST_DEVICE
    constexpr T& operator[](size_t i) {
        KMM_DEBUG_ASSERT(i < 1);
        return x;
    }

    KMM_HOST_DEVICE
    constexpr const T& operator[](size_t i) const {
        KMM_DEBUG_ASSERT(i < 1);
        return x;
    }

    T x;
};

template<typename T>
struct Vec<T, 2> {
    KMM_HOST_DEVICE
    constexpr Vec() {}

    KMM_HOST_DEVICE
    constexpr Vec(const T& x, const T& y) : x(x), y(y) {}

    KMM_HOST_DEVICE
    constexpr T& operator[](size_t i) {
        KMM_DEBUG_ASSERT(i < 2);
        constexpr decltype(&Vec::x) members[] = {&Vec::x, &Vec::y};
        return this->*members[i];
    }

    KMM_HOST_DEVICE
    constexpr const T& operator[](size_t i) const {
        KMM_DEBUG_ASSERT(i < 2);
        constexpr decltype(&Vec::x) members[] = {&Vec::x, &Vec::y};
        return this->*members[i];
    }

    T x;
    T y;
};

template<typename T>
struct Vec<T, 3> {
    KMM_HOST_DEVICE
    constexpr Vec() {}

    KMM_HOST_DEVICE
    constexpr Vec(const T& x, const T& y, const T& z) : x(x), y(y), z(z) {}

    KMM_HOST_DEVICE
    constexpr T& operator[](size_t i) {
        KMM_DEBUG_ASSERT(i < 3);
        constexpr decltype(&Vec::x) members[] = {&Vec::x, &Vec::y, &Vec::z};
        return this->*members[i];
    }

    KMM_HOST_DEVICE
    constexpr const T& operator[](size_t i) const {
        KMM_DEBUG_ASSERT(i < 3);
        constexpr decltype(&Vec::x) members[] = {&Vec::x, &Vec::y, &Vec::z};
        return this->*members[i];
    }

    T x;
    T y;
    T z;
};

template<typename T>
struct Vec<T, 4> {
    KMM_HOST_DEVICE
    constexpr Vec() {}

    KMM_HOST_DEVICE
    constexpr Vec(const T& x, const T& y, const T& z, const T& w) : x(x), y(y), z(z), w(w) {}

    KMM_HOST_DEVICE
    constexpr T& operator[](size_t i) {
        KMM_DEBUG_ASSERT(i < 4);
        constexpr decltype(&Vec::x) members[] = {&Vec::x, &Vec::y, &Vec::z, &Vec::w};
        return this->*members[i];
    }

    KMM_HOST_DEVICE
    constexpr const T& operator[](size_t i) const {
        KMM_DEBUG_ASSERT(i < 4);
        constexpr decltype(&Vec::x) members[] = {&Vec::x, &Vec::y, &Vec::z, &Vec::w};
        return this->*members[i];
    }

    T x;
    T y;
    T z;
    T w;
};

template<typename T>
Vec(const T&) -> Vec<T, 1>;

template<typename T>
Vec(const T&, const T&) -> Vec<T, 2>;

template<typename T>
Vec(const T&, const T&, const T&) -> Vec<T, 3>;

template<typename T>
Vec(const T&, const T&, const T&, const T&) -> Vec<T, 4>;

template<size_t N, typename T>
KMM_HOST_DEVICE constexpr Vec<T, N> fill(const T& value) {
    return make_index_sequence<N>::template fill<Vec<T, N>>(value);
}

/// @}

namespace detail {
template<typename T, size_t N, size_t M, size_t... Is, size_t... Js>
KMM_HOST_DEVICE Vec<T, N + M> concat_impl(
    const Vec<T, N>& lhs,
    const Vec<T, M>& rhs,
    IndexSequence<Is...>,
    IndexSequence<Js...>
) {
    return {lhs[Is]..., rhs[Js]...};
}
}  // namespace detail

/// \addtogroup geometry
/// @{

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Vec<T, N + M> concat(const Vec<T, N>& lhs, const Vec<T, M>& rhs) {
    return detail::concat_impl(lhs, rhs, make_index_sequence<N>(), make_index_sequence<M>());
}

template<typename T, typename U, size_t N>
KMM_HOST_DEVICE bool operator==(const Vec<T, N>& lhs, const Vec<U, N>& rhs) {
    bool result = true;

    if constexpr (N > 0) {
        for (size_t i = 0; is_less(i, N); i++) {
            result &= is_equal(lhs[i], rhs[i]);
        }
    }

    return result;
}

template<typename T, typename U, size_t N>
KMM_HOST_DEVICE bool operator!=(const Vec<T, N>& lhs, const Vec<U, N>& rhs) {
    return !(lhs == rhs);
}

/// @}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {
template<typename T, size_t N>
std::ostream& operator<<(std::ostream& stream, const Vec<T, N>& p) {
    stream << "{";
    if constexpr (N > 0) {
        stream << p[0];

        for (size_t i = 1; is_less(i, N); i++) {
            stream << ", " << p[i];
        }
    }
    return stream << "}";
}
}  // namespace kmm

template<typename T, size_t N>
struct fmt::formatter<kmm::Vec<T, N>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::Vec<T, N>> {
    size_t operator()(const kmm::Vec<T, N>& p) const {
        size_t result = 0;
        for (size_t i = 0; kmm::is_less(i, N); i++) {
            kmm::hash_combine(result, p[i]);
        }
        return result;
    }
};
#endif