#pragma once

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"

namespace kmm {

namespace detail {

enum class numeric_type_tag {  //
    signed_int,
    unsigned_int,
    floating_point,
    other
};

template<typename T>
struct numeric_type_traits {
    static constexpr numeric_type_tag tag = numeric_type_tag::other;
};

template<>
struct numeric_type_traits<float> {
    static constexpr numeric_type_tag tag = numeric_type_tag::floating_point;

    KMM_HOST_DEVICE
    static constexpr bool isnan(float f) {
        return f != f;
    }

    KMM_HOST_DEVICE
    static float ceil(float f) {
        return ::ceilf(f);
    }

    KMM_HOST_DEVICE
    static float floor(float f) {
        return ::floorf(f);
    }
};

template<>
struct numeric_type_traits<double> {
    static constexpr numeric_type_tag tag = numeric_type_tag::floating_point;

    KMM_HOST_DEVICE
    static constexpr bool isnan(double f) {
        return f != f;
    }

    KMM_HOST_DEVICE
    static double ceil(double f) {
        return ::ceil(f);
    }

    KMM_HOST_DEVICE
    static double floor(double f) {
        return ::floor(f);
    }
};

template<>
struct numeric_type_traits<bool> {
    static constexpr bool is_signed = false;
    using unsigned_type = bool;

    static constexpr numeric_type_tag tag = numeric_type_tag::unsigned_int;
    static constexpr bool min_inclusive = false;
    static constexpr bool max_inclusive = true;

    static constexpr float min_inclusive_float = 0.0F;
    static constexpr float max_exclusive_float = 2.0F;
};

#define KMM_DEFINE_INT_TRAITS(T)                                                          \
    template<>                                                                            \
    struct numeric_type_traits<signed T> {                                                \
        static constexpr bool is_signed = true;                                           \
        static constexpr numeric_type_tag tag = numeric_type_tag::signed_int;             \
        using unsigned_type = unsigned T;                                                 \
                                                                                          \
        static constexpr signed T max_inclusive =                                         \
            (signed T)(unsigned_type(~unsigned_type(0)) >> 1);                            \
        static constexpr signed T min_inclusive = ~max_inclusive;                         \
                                                                                          \
        static constexpr float min_inclusive_float = min_inclusive;                       \
        static constexpr float max_exclusive_float = unsigned_type(max_inclusive) + 1;    \
    };                                                                                    \
                                                                                          \
    template<>                                                                            \
    struct numeric_type_traits<unsigned T> {                                              \
        static constexpr bool is_signed = false;                                          \
        static constexpr numeric_type_tag tag = numeric_type_tag::unsigned_int;           \
                                                                                          \
        static constexpr unsigned T min_inclusive = 0;                                    \
        static constexpr unsigned T max_inclusive = ~static_cast<unsigned T>(0);          \
                                                                                          \
        static constexpr float min_inclusive_float = min_inclusive;                       \
        static constexpr float max_exclusive_float = 2.0f * float(max_inclusive / 2 + 1); \
    };

KMM_DEFINE_INT_TRAITS(char)
KMM_DEFINE_INT_TRAITS(short)
KMM_DEFINE_INT_TRAITS(int)
KMM_DEFINE_INT_TRAITS(long)
KMM_DEFINE_INT_TRAITS(long long)

template<typename L, typename R>
struct checked_compare_base {
    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return left == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return left < right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less_equal(L left, R right) {
        return left <= right;
    }
};

template<
    typename L,
    typename R,
    numeric_type_tag = numeric_type_traits<L>::tag,
    numeric_type_tag = numeric_type_traits<R>::tag>
struct checked_compare_impl;

template<typename T>
struct checked_compare_impl<T, T, numeric_type_tag::other, numeric_type_tag::other>:
    checked_compare_base<T, T> {};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::unsigned_int, numeric_type_tag::unsigned_int>:
    checked_compare_base<L, R> {};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::signed_int, numeric_type_tag::signed_int>:
    checked_compare_base<L, R> {};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::unsigned_int, numeric_type_tag::signed_int> {
    using UR = typename numeric_type_traits<R>::unsigned_type;

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return right >= static_cast<R>(0) && left == static_cast<UR>(right);
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return right >= static_cast<R>(0) && left < static_cast<UR>(right);
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less_equal(L left, R right) {
        return right >= static_cast<R>(0) && left <= static_cast<UR>(right);
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::signed_int, numeric_type_tag::unsigned_int> {
    using UL = typename numeric_type_traits<L>::unsigned_type;

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return left >= static_cast<L>(0) && static_cast<UL>(left) == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return left < static_cast<L>(0) || static_cast<UL>(left) < right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less_equal(L left, R right) {
        return left < static_cast<L>(0) || static_cast<UL>(left) <= right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<
    L,
    R,
    numeric_type_tag::floating_point,
    numeric_type_tag::floating_point>: checked_compare_base<L, R> {};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::floating_point, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (numeric_type_traits<L>::isnan(left)) {
            return false;
        }

        if (numeric_type_traits<L>::floor(left) < numeric_type_traits<R>::min_inclusive_float) {
            return true;
        }

        if (!(numeric_type_traits<L>::floor(left) < numeric_type_traits<R>::max_exclusive_float)) {
            return false;
        }

        return static_cast<R>(numeric_type_traits<L>::floor(left)) < right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        if (numeric_type_traits<L>::floor(left) != left) {
            return false;
        }

        if (left < numeric_type_traits<R>::min_inclusive_float) {
            return false;
        }

        if (!(left < numeric_type_traits<R>::max_exclusive_float)) {
            return false;
        }

        return static_cast<R>(left) == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less_equal(L left, R right) {
        return is_less(left, right) || is_equal(left, right);
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::signed_int, numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (numeric_type_traits<R>::isnan(right)) {
            return false;
        }

        if (numeric_type_traits<R>::ceil(right) < numeric_type_traits<L>::min_inclusive_float) {
            return false;
        }

        if (!(numeric_type_traits<R>::ceil(right) < numeric_type_traits<L>::max_exclusive_float)) {
            return true;
        }

        return left < static_cast<L>(numeric_type_traits<R>::ceil(right));
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return checked_compare_impl<R, L>::is_equal(right, left);
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less_equal(L left, R right) {
        return is_less(left, right) || is_equal(left, right);
    }
};

template<typename L, typename R>
struct checked_compare_impl<
    L,
    R,
    numeric_type_tag::floating_point,
    numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (numeric_type_traits<L>::isnan(left)) {
            return false;
        }

        if (numeric_type_traits<L>::floor(left) < numeric_type_traits<R>::min_inclusive_float) {
            return true;
        }

        if (!(numeric_type_traits<L>::floor(left) < numeric_type_traits<R>::max_exclusive_float)) {
            return false;
        }

        return static_cast<R>(numeric_type_traits<L>::floor(left)) < right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        if (numeric_type_traits<L>::isnan(left)) {
            return false;
        }

        if (numeric_type_traits<L>::floor(left) != left) {
            return false;
        }

        if (left < numeric_type_traits<R>::min_inclusive_float) {
            return false;
        }

        if (!(left < numeric_type_traits<R>::max_exclusive_float)) {
            return false;
        }

        return static_cast<R>(left) == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less_equal(L left, R right) {
        return is_less(left, right) || is_equal(left, right);
    }
};

template<typename L, typename R>
struct checked_compare_impl<
    L,
    R,
    numeric_type_tag::unsigned_int,
    numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (numeric_type_traits<R>::isnan(right)) {
            return false;
        }

        if (numeric_type_traits<R>::ceil(right) < numeric_type_traits<L>::min_inclusive_float) {
            return false;
        }

        if (!(numeric_type_traits<R>::ceil(right) < numeric_type_traits<L>::max_exclusive_float)) {
            return true;
        }

        return left < static_cast<L>(numeric_type_traits<R>::ceil(right));
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return checked_compare_impl<R, L>::is_equal(right, left);
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less_equal(L left, R right) {
        return is_less(left, right) || is_equal(left, right);
    }
};

template<
    typename I,
    typename O,
    numeric_type_tag = numeric_type_traits<I>::tag,
    numeric_type_tag = numeric_type_traits<O>::tag>
struct is_convertible_impl;

template<typename T>
struct is_convertible_impl<T, T, numeric_type_tag::other, numeric_type_tag::other> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const T& input) {
        return true;
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::signed_int, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        return input >= numeric_type_traits<O>::min_inclusive
            && input <= numeric_type_traits<O>::max_inclusive;
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::unsigned_int, numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        return input <= numeric_type_traits<O>::max_inclusive;
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::unsigned_int, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        using UO = typename numeric_type_traits<O>::unsigned_type;
        return input <= static_cast<UO>(numeric_type_traits<O>::max_inclusive);
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::signed_int, numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        using UI = typename numeric_type_traits<I>::unsigned_type;
        return input >= static_cast<I>(0)
            && is_convertible_impl<UI, O>::apply(static_cast<UI>(input));
    }
};

template<typename I, typename O>
struct is_convertible_impl<
    I,
    O,
    numeric_type_tag::floating_point,
    numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        return input == static_cast<O>(input);
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::floating_point, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        if (input < numeric_type_traits<O>::min_inclusive_float) {
            return false;
        }

        if (!(input < numeric_type_traits<O>::max_exclusive_float)) {
            return false;
        }

        return static_cast<I>(static_cast<O>(input)) == input;
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::floating_point, numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        if (input < static_cast<I>(0)) {
            return false;
        }

        if (!(input < numeric_type_traits<O>::max_exclusive_float)) {
            return false;
        }

        return static_cast<I>(static_cast<O>(input)) == input;
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::signed_int, numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        return checked_compare_impl<I, O>::is_equal(input, static_cast<O>(input));
    }
};

template<typename I, typename O>
struct is_convertible_impl<I, O, numeric_type_tag::unsigned_int, numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input) {
        return checked_compare_impl<I, O>::is_equal(input, static_cast<O>(input));
    }
};

template<typename I, typename O>
struct checked_cast_impl {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input, O* output) {
        if (!is_convertible_impl<I, O>::apply(input)) {
            return false;
        }

        *output = static_cast<O>(input);
        return true;
    }
};
}  // namespace detail

#if KMM_IS_DEVICE
// on the GPU, we just panic immediately
KMM_DEVICE void throw_overflow_exception() {
    KMM_PANIC("overflow occurred in operation");
}
#else
// on the host, we can throw an exception
[[noreturn]] void throw_overflow_exception();
#endif

/// \addtogroup utility
/// @{

/**
 * All functions below (`is_less`, `is_equal`, etc.) cast their arguments to the underlying types
 * before performing the operation. By default, the underlying type of `T` is just `T`.
 *
 * However, other types can override this by specializing `underlying_type<T>` for their own
 * type and setting `type` to whatever native type it should be compared/converted as.
 */
template<typename T>
struct underlying_type {
    using type = T;
};

template<typename T>
using underlying_type_t = typename underlying_type<T>::type;

/// Returns whether `left < right`, safely comparing operands of different types and signedness.
template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_less(L left, R right) {
    return detail::checked_compare_impl<underlying_type_t<L>, underlying_type_t<R>>::is_less(
        static_cast<underlying_type_t<L>>(left),
        static_cast<underlying_type_t<R>>(right)
    );
}

/// Returns whether `left == right`, safely comparing operands of different types and signedness.
template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_equal(L left, R right) {
    return detail::checked_compare_impl<underlying_type_t<L>, underlying_type_t<R>>::is_equal(
        static_cast<underlying_type_t<L>>(left),
        static_cast<underlying_type_t<R>>(right)
    );
}

/// Returns whether `left <= right`, safely comparing operands of different types and signedness.
template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_less_equal(L left, R right) {
    return detail::checked_compare_impl<underlying_type_t<L>, underlying_type_t<R>>::is_less_equal(
        static_cast<underlying_type_t<L>>(left),
        static_cast<underlying_type_t<R>>(right)
    );
}

/// Returns whether `left > right`, safely comparing operands of different types and signedness.
template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_greater(L left, R right) {
    return is_less(right, left);
}

/// Returns whether `left >= right`, safely comparing operands of different types and signedness.
template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_greater_equal(L left, R right) {
    return is_less_equal(right, left);
}

/// Returns `true` if the given `input` value can safely be converted from type `T` to type `U`,
/// and `false` otherwise. For example, `int32_t(100)` can be converted to `uint16_t`, but
/// `int32_t(-100)` cannot since it cannot be represented as a `uint16_t`.
template<typename U, typename T>
KMM_HOST_DEVICE constexpr bool is_convertible(const T& input) {
    U output {};
    return detail::checked_cast_impl<underlying_type_t<T>, U>::apply(
        static_cast<underlying_type_t<T>>(input),
        &output
    );
}

/// Casts the given value `input` from type `T` to type `U`. Throws an exception if the
/// input value cannot be represented as `U`.
template<typename U, typename T>
KMM_HOST_DEVICE constexpr U checked_cast(const T& input) {
    U output {};

    if (!detail::checked_cast_impl<underlying_type_t<T>, U>::apply(
            static_cast<underlying_type_t<T>>(input),
            &output
        )) {
        throw_overflow_exception();
    }

    return output;
}

/// @}

}  // namespace kmm