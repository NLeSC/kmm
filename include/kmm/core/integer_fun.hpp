#pragma once

#include "kmm/core/checked_math.hpp"
#include "kmm/core/macros.hpp"

namespace kmm {

/// \addtogroup utility
/// @{

/// Divide `num` by `denom` and round the result down (towards negative infinity).
template<typename T>
KMM_HOST_DEVICE constexpr T div_floor(T a, T b) {
    const T zero = static_cast<T>(0);
    T quotient = a / b;

    if constexpr (detail::numeric_type_traits<T>::is_signed) {
        // Adjust the quotient if a and b have different signs
        if (a % b != zero && ((a >= zero) ^ (b >= zero))) {
            quotient -= 1;
        }
    }

    return quotient;
}

/// Divide `num` by `denom` and round the result up (towards positive infinity).
template<typename T>
KMM_HOST_DEVICE constexpr T div_ceil(T a, T b) {
    const T zero = static_cast<T>(0);
    T quotient = a / b;

    if constexpr (detail::numeric_type_traits<T>::is_signed) {
        // Adjust the quotient if both a and b have the same sign
        if (a % b != zero && !((a >= zero) ^ (b >= zero))) {
            quotient += 1;
        }
    } else if (a % b != zero) {
        quotient += 1;
    }

    return quotient;
}

/// Round `input` to the next multiple of `multiple`.
///
/// This returns the smallest value greater than or equal to `input` that is divisible
/// by `multiple`.
template<typename T>
KMM_HOST_DEVICE constexpr T round_up_to_multiple(T input, T multiple) {
    const T zero = static_cast<T>(0);
    T remainder = input % multiple;
    T delta = 0;

    if (remainder != zero) {
        if constexpr (detail::numeric_type_traits<T>::is_signed) {
            // There are two cases:
            // - input >= 0: return input + abs(multiple) - remainder
            // - input < 0:  return input - remainder
            T abs_multiple = (multiple >= zero ? multiple : -multiple);
            delta = abs_multiple * (input >= zero) - remainder;
        } else {
            delta = multiple - remainder;
        }
    }

    return checked_add(input, delta);
}

/// Return the absolute value of `input` as the corresponding unsigned integer type.
template<typename T, typename U = typename detail::numeric_type_traits<T>::unsigned_type>
KMM_HOST_DEVICE constexpr U unsigned_abs(T input) {
    U magnitude = static_cast<U>(input);
    return detail::is_negative(input) ? static_cast<U>(U(0) - magnitude) : magnitude;
}

/// Returns `true` if `left` is exactly divisible by `right` (i.e. `left % right == 0`).
/// Returns `false` otherwise. This function accepts mixed inputs having different signedness.
template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_divisible(L left, R right) {
    return right != R {0} && unsigned_abs(left) % unsigned_abs(right) == 0U;
}

/// Return the smallest integer that is a power of two and is not less than `input`.
template<typename T>
KMM_HOST_DEVICE constexpr T round_up_to_power_of_two(T input) {
    if (input <= static_cast<T>(0)) {
        return static_cast<T>(1);
    }

    input -= static_cast<T>(1);
    for (decltype(sizeof(T)) i = 1; i < sizeof(T) * 8; i *= 2) {
        input |= (input >> i);
    }

    return checked_add(input, static_cast<T>(1));
}

/// @}

}  // namespace kmm