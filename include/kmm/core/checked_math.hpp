#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/macros.hpp"

// __mul64hi/__umul64hi are bare compiler builtins under CUDA, but HIP only declares them once its
// runtime header is included.
#if KMM_IS_DEVICE
    #if defined(__CUDACC__)
        #include <cuda_runtime.h>
    #elif defined(__HIPCC__)
        #include <hip/hip_runtime.h>
    #endif
#endif

namespace kmm {

namespace detail {

template<typename T, numeric_type_tag = numeric_type_traits<T>::tag>
struct wider_type;

template<typename T>
struct wider_type<T, numeric_type_tag::signed_int> {
    using type = int64_t;
};

template<typename T>
struct wider_type<T, numeric_type_tag::unsigned_int> {
    using type = uint64_t;
};

template<typename T>
using wider_t = typename wider_type<T>::type;

// `value < 0` guarded so it never becomes a "comparison of unsigned with zero" for unsigned `T`.
template<typename T>
KMM_HOST_DEVICE constexpr bool is_negative(T value) {
    if constexpr (numeric_type_traits<T>::tag == numeric_type_tag::signed_int) {
        return value < static_cast<T>(0);
    } else {
        return false;
    }
}

template<typename L, typename R, typename O>
struct checked_add_impl {
    KMM_HOST_DEVICE
    static constexpr bool apply(L left, R right, O* output) {
        uint64_t sum = static_cast<uint64_t>(static_cast<wider_t<L>>(left))
            + static_cast<uint64_t>(static_cast<wider_t<R>>(right));
        *output = static_cast<O>(sum);

        bool left_negative = is_negative(left);
        bool right_negative = is_negative(right);
        bool carry = sum < static_cast<uint64_t>(static_cast<wider_t<L>>(left));
        bool sum_negative = numeric_type_traits<O>::is_signed && (static_cast<int64_t>(sum) < 0);

        bool is_valid = int(carry) - int(left_negative) - int(right_negative) == -int(sum_negative);

        // if all are signed, the above can be simplified to just this:
        // same-sign operands overflow iff the result's sign differs from both.
        if constexpr (
            numeric_type_traits<L>::is_signed &&  //
            numeric_type_traits<R>::is_signed &&  //
            numeric_type_traits<O>::is_signed
        ) {
            int64_t l = static_cast<int64_t>(left);
            int64_t r = static_cast<int64_t>(right);
            int64_t s = static_cast<int64_t>(sum);
            is_valid = ((s ^ l) & (s ^ r)) >= 0;
        }

        return is_valid && is_convertible_impl<wider_t<O>, O>::apply(static_cast<wider_t<O>>(sum));
    }
};

template<typename L, typename R, typename O>
struct checked_sub_impl {
    KMM_HOST_DEVICE
    static constexpr bool apply(L left, R right, O* output) {
        uint64_t diff = static_cast<uint64_t>(static_cast<wider_t<L>>(left))
            - static_cast<uint64_t>(static_cast<wider_t<R>>(right));
        *output = static_cast<O>(diff);

        bool left_negative = is_negative(left);
        bool right_negative = is_negative(right);
        bool borrow = static_cast<uint64_t>(static_cast<wider_t<L>>(left))
            < static_cast<uint64_t>(static_cast<wider_t<R>>(right));
        bool diff_negative = numeric_type_traits<O>::is_signed && static_cast<int64_t>(diff) < 0;

        bool is_valid =
            int(borrow) - int(right_negative) - int(diff_negative) == -int(left_negative);

        // if all are signed, the above can be simplified to just this:
        // same-sign operands never overflow; for mixed signs, overflow iff the result's
        // sign differs from `left`.
        if constexpr (
            numeric_type_traits<L>::is_signed &&  //
            numeric_type_traits<R>::is_signed &&  //
            numeric_type_traits<O>::is_signed
        ) {
            int64_t l = static_cast<int64_t>(left);
            int64_t r = static_cast<int64_t>(right);
            int64_t d = static_cast<int64_t>(diff);
            is_valid = ((l ^ r) & (l ^ d)) >= 0;
        }

        return is_valid && is_convertible_impl<wider_t<O>, O>::apply(static_cast<wider_t<O>>(diff));
    }
};

template<typename L, typename R, typename O>
struct checked_mul_impl {
    KMM_HOST_DEVICE
    static constexpr bool apply(L left, R right, O* output) {
#if KMM_IS_DEVICE
        // Fast path when every operand is signed
        if constexpr (
            numeric_type_traits<L>::is_signed &&  //
            numeric_type_traits<R>::is_signed &&  //
            numeric_type_traits<O>::is_signed
        ) {
            int64_t l = static_cast<int64_t>(left);
            int64_t r = static_cast<int64_t>(right);
            int64_t lo = static_cast<int64_t>(static_cast<uint64_t>(l) * static_cast<uint64_t>(r));

            // must have same sign.
            if (__mul64hi(l, r) != (lo >> 63)) {
                return false;
            }

            *output = static_cast<O>(lo);
            return is_convertible_impl<int64_t, O>::apply(lo);
        }

        // General path (mixed signedness): reduce both operands to unsigned magnitudes, use the
        // hardware high-multiply to detect a product wider than 64 bits, then reapply the sign.
        uint64_t al = static_cast<uint64_t>(static_cast<wider_t<L>>(left));
        uint64_t ar = static_cast<uint64_t>(static_cast<wider_t<R>>(right));

        if (is_negative(left)) {
            al = -al;
        }

        if (is_negative(right)) {
            ar = -ar;
        }

        if (__umul64hi(al, ar) != 0) {
            return false;
        }

        uint64_t magnitude = al * ar;
        bool result_negative = is_negative(left) != is_negative(right);

        if (result_negative) {
            return checked_sub_impl<uint64_t, uint64_t, O>::apply(uint64_t(0), magnitude, output);
        } else {
            return checked_add_impl<uint64_t, uint64_t, O>::apply(uint64_t(0), magnitude, output);
        }
#else
        return !__builtin_mul_overflow(left, right, output);
#endif
    }
};

template<typename L, typename R, typename O>
struct checked_div_impl {
    KMM_HOST_DEVICE
    static constexpr bool apply(L left, R right, O* output) {
        // division by zero
        if (right == static_cast<R>(0)) {
            return false;
        }

        // If both are signed, the result fits into int64_t, except for MIN/-1
        if constexpr (numeric_type_traits<L>::is_signed && numeric_type_traits<R>::is_signed) {
            if (left == numeric_type_traits<int64_t>::min_inclusive && right == -1) {
                uint64_t magnitude = uint64_t(numeric_type_traits<int64_t>::max_inclusive) + 1;
                *output = static_cast<O>(magnitude);
                return is_convertible_impl<uint64_t, O>::apply(magnitude);
            } else {
                int64_t magnitude = static_cast<int64_t>(left) / static_cast<int64_t>(right);
                *output = static_cast<O>(magnitude);
                return is_convertible_impl<int64_t, O>::apply(magnitude);
            }
        }

        uint64_t al = static_cast<uint64_t>(static_cast<wider_t<L>>(left));
        uint64_t ar = static_cast<uint64_t>(static_cast<wider_t<R>>(right));

        if (is_negative(left)) {
            al = -al;
        }

        if (is_negative(right)) {
            ar = -ar;
        }

        uint64_t magnitude = al / ar;
        bool result_negative = is_negative(left) != is_negative(right);

        if (result_negative) {
            return checked_sub_impl<uint64_t, uint64_t, O>::apply(uint64_t(0), magnitude, output);
        } else {
            return checked_add_impl<uint64_t, uint64_t, O>::apply(uint64_t(0), magnitude, output);
        }
    }
};

template<typename L, typename R, typename O>
struct checked_rem_impl {
    KMM_HOST_DEVICE
    static constexpr bool apply(L left, R right, O* output) {
        // division by zero
        if (right == static_cast<R>(0)) {
            return false;
        }

        // If both are signed, the result fits into int64_t, except for MIN/-1
        if constexpr (numeric_type_traits<L>::is_signed && numeric_type_traits<R>::is_signed) {
            int64_t magnitude;

            if (left == numeric_type_traits<int64_t>::min_inclusive && right == -1) {
                magnitude = static_cast<int64_t>(0);
            } else {
                magnitude = static_cast<int64_t>(left) % static_cast<int64_t>(right);
            }

            *output = static_cast<O>(magnitude);
            return is_convertible_impl<int64_t, O>::apply(magnitude);
        }

        uint64_t al = static_cast<uint64_t>(static_cast<wider_t<L>>(left));
        uint64_t ar = static_cast<uint64_t>(static_cast<wider_t<R>>(right));

        if (is_negative(left)) {
            al = -al;
        }

        if (is_negative(right)) {
            ar = -ar;
        }

        uint64_t magnitude = al % ar;
        bool result_negative = is_negative(left);  // sign follows dividend

        if (result_negative) {
            return checked_sub_impl<uint64_t, uint64_t, O>::apply(uint64_t(0), magnitude, output);
        } else {
            return checked_add_impl<uint64_t, uint64_t, O>::apply(uint64_t(0), magnitude, output);
        }
    }
};

// These `__builtin_*_overflow` compiler builtins have no device-side implementation, so these
// specializations are compiled for the host pass only; the device pass never sees them declared
// and falls back to the generic, non-specialized `checked_*_impl<L, R, O>::apply` above instead.
#if !KMM_IS_DEVICE
    #define KMM_CHECKED_MATH_IMPL(OP, T, FUN)               \
        template<>                                          \
        struct OP<T, T, T> {                                \
            KMM_HOST_DEVICE                                 \
            static bool apply(T left, T right, T* output) { \
                return FUN(left, right, output) == false;   \
            }                                               \
        };

KMM_CHECKED_MATH_IMPL(checked_add_impl, int, __builtin_sadd_overflow)
KMM_CHECKED_MATH_IMPL(checked_add_impl, long, __builtin_saddl_overflow)
KMM_CHECKED_MATH_IMPL(checked_add_impl, long long, __builtin_saddll_overflow)
KMM_CHECKED_MATH_IMPL(checked_add_impl, unsigned int, __builtin_uadd_overflow)
KMM_CHECKED_MATH_IMPL(checked_add_impl, unsigned long, __builtin_uaddl_overflow)
KMM_CHECKED_MATH_IMPL(checked_add_impl, unsigned long long, __builtin_uaddll_overflow)

KMM_CHECKED_MATH_IMPL(checked_sub_impl, int, __builtin_ssub_overflow)
KMM_CHECKED_MATH_IMPL(checked_sub_impl, long, __builtin_ssubl_overflow)
KMM_CHECKED_MATH_IMPL(checked_sub_impl, long long, __builtin_ssubll_overflow)
KMM_CHECKED_MATH_IMPL(checked_sub_impl, unsigned int, __builtin_usub_overflow)
KMM_CHECKED_MATH_IMPL(checked_sub_impl, unsigned long, __builtin_usubl_overflow)
KMM_CHECKED_MATH_IMPL(checked_sub_impl, unsigned long long, __builtin_usubll_overflow)

KMM_CHECKED_MATH_IMPL(checked_mul_impl, int, __builtin_smul_overflow)
KMM_CHECKED_MATH_IMPL(checked_mul_impl, long, __builtin_smull_overflow)
KMM_CHECKED_MATH_IMPL(checked_mul_impl, long long, __builtin_smulll_overflow)
KMM_CHECKED_MATH_IMPL(checked_mul_impl, unsigned int, __builtin_umul_overflow)
KMM_CHECKED_MATH_IMPL(checked_mul_impl, unsigned long, __builtin_umull_overflow)
KMM_CHECKED_MATH_IMPL(checked_mul_impl, unsigned long long, __builtin_umulll_overflow)
    #undef KMM_CHECKED_MATH_IMPL
#endif  // !KMM_IS_DEVICE

}  // namespace detail

/// \addtogroup utility
/// @{

/// Returns `left + right`, throwing on overflow.
template<typename O, typename L, typename R>
KMM_HOST_DEVICE constexpr O checked_add(L left, R right) {
    O output {};

    if (!detail::checked_add_impl<L, R, O>::apply(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<decltype(nullptr) = nullptr, typename T>
KMM_HOST_DEVICE constexpr T checked_add(T left, T right) {
    return checked_add<T, T, T>(left, right);
}

/// Returns `left - right`, throwing on overflow.
template<typename O, typename L, typename R>
KMM_HOST_DEVICE constexpr O checked_sub(L left, R right) {
    O output {};

    if (!detail::checked_sub_impl<L, R, O>::apply(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<decltype(nullptr) = nullptr, typename T>
KMM_HOST_DEVICE constexpr T checked_sub(T left, T right) {
    return checked_sub<T, T, T>(left, right);
}

/// Returns `left * right`, throwing on overflow.
template<typename O, typename L, typename R>
KMM_HOST_DEVICE constexpr O checked_mul(L left, R right) {
    O output {};

    if (!detail::checked_mul_impl<L, R, O>::apply(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<decltype(nullptr) = nullptr, typename T>
KMM_HOST_DEVICE constexpr T checked_mul(T left, T right) {
    return checked_mul<T, T, T>(left, right);
}

/// Returns `left / right`, throwing on division by zero or overflow.
template<typename O, typename L, typename R>
KMM_HOST_DEVICE constexpr O checked_div(L left, R right) {
    O output {};

    if (!detail::checked_div_impl<L, R, O>::apply(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<decltype(nullptr) = nullptr, typename T>
KMM_HOST_DEVICE constexpr T checked_div(T left, T right) {
    return checked_div<T, T, T>(left, right);
}

/// Returns `left % right`, throwing on division by zero or overflow.
template<typename O, typename L, typename R>
KMM_HOST_DEVICE constexpr O checked_rem(L left, R right) {
    O output {};

    if (!detail::checked_rem_impl<L, R, O>::apply(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<decltype(nullptr) = nullptr, typename T>
KMM_HOST_DEVICE constexpr T checked_rem(T left, T right) {
    return checked_rem<T, T, T>(left, right);
}

/// Returns `-input`, throwing on overflow.
template<typename T>
KMM_HOST_DEVICE constexpr T checked_neg(const T& input) {
    return checked_sub(static_cast<T>(0), input);
}

/// Returns `abs(input)`, throwing on overflow.
template<typename T>
KMM_HOST_DEVICE constexpr T checked_abs(const T& input) {
    return is_less(input, static_cast<T>(0)) ? checked_neg(input) : input;
}

/// Returns `begin[0] + begin[1] + ... + begin[end - begin]`, throwing on overflow.
template<typename It, typename U = decltype(*It() + *It())>
KMM_HOST_DEVICE U checked_sum(It begin, It end, U initial = U(0)) {
    using T = decltype(+*begin);
    U accum = initial;
    bool is_valid = true;

    for (It it = begin; it != end; ++it) {
        if (!detail::checked_add_impl<U, T, U>::apply(accum, *it, &accum)) {
            is_valid = false;
        }
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return accum;
}

/// Returns `begin[0] * begin[1] * ... * begin[end - begin]`, throwing on overflow.
template<typename It, typename U = decltype(*It() * *It())>
KMM_HOST_DEVICE U checked_product(It begin, It end, U initial = U(1)) {
    using T = decltype(+*begin);
    bool is_valid = true;
    U accum = initial;

    for (It it = begin; it != end; ++it) {
        if (!detail::checked_mul_impl<U, T, U>::apply(accum, *it, &accum)) {
            is_valid = false;
        }
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return accum;
}

/// @}

}  // namespace kmm
