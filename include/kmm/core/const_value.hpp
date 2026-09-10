#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/checked_math.hpp"
#include "kmm/core/macros.hpp"

namespace kmm {

namespace detail {
template<auto OtherValue, typename value_type, bool Cond = is_convertible<value_type>(OtherValue)>
struct is_convertible_helper {};

template<auto OtherValue, typename value_type>
struct is_convertible_helper<OtherValue, value_type, true> {
    using type = void;
};
}  // namespace detail

/// Represents a value that is known at compile time.
template<auto Value>
struct ConstValue {
    using type = ConstValue;
    using value_type = decltype(Value);
    static constexpr value_type value = Value;

    KMM_HOST_DEVICE
    constexpr ConstValue() = default;

    template<
        auto OtherValue,
        typename = typename detail::is_convertible_helper<OtherValue, value_type>::type>
    KMM_HOST_DEVICE constexpr ConstValue(ConstValue<OtherValue>) noexcept {}

    KMM_HOST_DEVICE
    constexpr operator value_type() const noexcept {
        return Value;
    }

    KMM_HOST_DEVICE
    constexpr value_type operator()() const noexcept {
        return value;
    }
};

template<decltype(sizeof(int)) I>
using ConstIndex = ConstValue<I>;

template<bool C>
using ConstBool = ConstValue<C>;

template<auto Value>
ConstValue<+Value> operator+(ConstValue<Value>) {
    return {};
}

template<auto Value>
ConstValue<-Value> operator-(ConstValue<Value>) {
    checked_neg(Value);
    return {};
}

#define KMM_CONST_VALUE_OP(OP, FUN)                                              \
    template<auto Left, auto Right>                                              \
    ConstValue<Left OP Right> operator OP(ConstValue<Left>, ConstValue<Right>) { \
        FUN(Left, Right);                                                        \
        return {};                                                               \
    }

KMM_CONST_VALUE_OP(+, checked_add)
KMM_CONST_VALUE_OP(-, checked_sub)
KMM_CONST_VALUE_OP(*, checked_mul)
KMM_CONST_VALUE_OP(/, checked_div)

#undef KMM_CONST_VALUE_OP

// ConstValue<Value> should compare/convert exactly like a plain `decltype(Value)`.
template<auto Value>
struct underlying_type<ConstValue<Value>>: underlying_type<decltype(Value)> {};

namespace detail {

// allows for `checked_cast<ConstValue<5>>(x)`.
template<typename I, auto Value>
struct checked_cast_impl<I, ConstValue<Value>> {
    KMM_HOST_DEVICE
    static constexpr bool apply(const I& input, ConstValue<Value>* output) {
        return is_equal(input, Value);
    }
};

}  // namespace detail

}  // namespace kmm
