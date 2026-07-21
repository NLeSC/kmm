#pragma once

#include "kmm/utils/checked_compare.hpp"
#include "kmm/utils/fixed_vector.hpp"
#include "kmm/utils/macros.hpp"
#include "kmm/utils/point.hpp"

namespace kmm {

template<size_t N, typename T = default_index_type>
struct Dim: public fixed_vector<T, N> {
  public:
    using storage_type = fixed_vector<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Dim(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Dim() {
        for (size_t i = 0; is_less(i, N); i++) {
            (*this)[i] = static_cast<T>(1);
        }
    }

    constexpr Dim(const Dim&) = default;
    constexpr Dim(Dim&&) noexcept = default;
    Dim& operator=(const Dim&) = default;
    Dim& operator=(Dim&&) noexcept = default;

    template<typename... Ts, typename = typename detail::enable_if<(sizeof...(Ts) + 1 == N)>::type>
    KMM_HOST_DEVICE Dim(T first, Ts&&... args) : Dim() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE constexpr Dim(const Dim<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Dim::from(that);
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Dim from(const fixed_vector<U, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? static_cast<T>(that[i]) : static_cast<T>(1);
        }

        return Dim(result);
    }

    KMM_HOST_DEVICE
    static Dim fill(T value) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = value;
        }

        return Dim(result);
    }

    KMM_HOST_DEVICE
    static Dim one() {
        return fill(static_cast<T>(1));
    }

    KMM_HOST_DEVICE
    static Dim zero() {
        return fill(static_cast<T>(0));
    }

    KMM_HOST_DEVICE
    T get_or_default(size_t i, T default_value = static_cast<T>(1)) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(is_less(i, N))) {
                return (*this)[i];
            }
        }

        return default_value;
    }

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

    KMM_HOST_DEVICE
    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= !(static_cast<T>(0) < (*this)[i]);
        }

        return result;
    }

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
Dim(Ts&&...) -> Dim<sizeof...(Ts)>;

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Dim<N + M, T> concat(const Dim<N, T>& lhs, const Dim<M, T>& rhs) {
    return Dim<N + M, T> {concat((const fixed_vector<T, N>&)(lhs), (const fixed_vector<T, M>&)(rhs))
    };
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator==(const Dim<N, T>& lhs, const Dim<M, U>& rhs) {
    bool result = true;

    for (size_t i = 0; is_less(i, N) || is_less(i, M); i++) {
        result &= is_equal(lhs.get_or_default(i), rhs.get_or_default(i));
    }

    return result;
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator!=(const Dim<N, T>& lhs, const Dim<M, U>& rhs) {
    return !(lhs == rhs);
}

namespace detail {
// Specialize comparison between Dim<T> and T
template<typename L, typename R, numeric_type_tag Ltag, numeric_type_tag Rtag>
struct checked_compare_impl<L, Dim<1, R>, Ltag, Rtag>: checked_compare_impl<L, R> {};

template<typename L, typename R, numeric_type_tag Ltag, numeric_type_tag Rtag>
struct checked_compare_impl<Dim<1, L>, R, Ltag, Rtag>: checked_compare_impl<L, R> {};

template<typename L, typename R, numeric_type_tag Ltag, numeric_type_tag Rtag>
struct checked_compare_impl<Dim<1, L>, Dim<1, R>, Ltag, Rtag>: checked_compare_impl<L, R> {};

template<typename T>
struct checked_compare_impl<Dim<1, T>, Dim<1, T>, numeric_type_tag::other, numeric_type_tag::other>:
    checked_compare_impl<T, T> {};

template<typename L, typename R, numeric_type_tag Ltag, numeric_type_tag Rtag>
struct checked_convert_impl<L, Dim<1, R>, Ltag, Rtag>: checked_convert_impl<L, R> {};

template<typename L, typename R, numeric_type_tag Ltag, numeric_type_tag Rtag>
struct checked_convert_impl<Dim<1, L>, R, Ltag, Rtag>: checked_convert_impl<L, R> {};

template<typename L, typename R, numeric_type_tag Ltag, numeric_type_tag Rtag>
struct checked_convert_impl<Dim<1, L>, Dim<1, R>, Ltag, Rtag>: checked_convert_impl<L, R> {};

template<typename T>
struct checked_convert_impl<Dim<1, T>, Dim<1, T>, numeric_type_tag::other, numeric_type_tag::other>:
    checked_convert_impl<T, T> {};
}  // namespace detail

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {
template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Dim<N, T>& p) {
    return stream << static_cast<const fixed_vector<T, N>&>(p);
}
}  // namespace kmm

template<size_t N, typename T>
struct fmt::formatter<kmm::Dim<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::Dim<N, T>>: std::hash<kmm::fixed_vector<T, N>> {};
#endif