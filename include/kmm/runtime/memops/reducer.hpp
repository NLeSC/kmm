#pragma once

#include <limits>
#include <type_traits>

#include "kmm/core/key_value.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/runtime/memops/types.hpp"

namespace kmm::memops {

/// Per-`ReductionOp` accumulator
template<typename T, ReductionOp Op, typename = void>
struct Reducer;

template<typename T>
struct Reducer<T, ReductionOp::Sum, std::enable_if_t<std::is_arithmetic_v<T>>> {
    using element_type = T;

    KMM_HOST_DEVICE void consume(T x) {
        value += x;
    }

    KMM_HOST_DEVICE T finish() {
        return value;
    }

    T value = 0;
};

template<typename T>
struct Reducer<T, ReductionOp::Product, std::enable_if_t<std::is_arithmetic_v<T>>> {
    using element_type = T;

    KMM_HOST_DEVICE void consume(T x) {
        value *= x;
    }

    KMM_HOST_DEVICE T finish() {
        return value;
    }

    T value = 1;
};

template<typename T>
struct Reducer<T, ReductionOp::Min, std::enable_if_t<std::is_arithmetic_v<T>>> {
    using element_type = T;

    KMM_HOST_DEVICE void consume(T x) {
        value = value < x ? value : x;
    }

    KMM_HOST_DEVICE T finish() {
        return value;
    }

    static constexpr T identity_value = std::numeric_limits<T>::max();
    T value = identity_value;
};

template<typename T>
struct Reducer<T, ReductionOp::Max, std::enable_if_t<std::is_arithmetic_v<T>>> {
    using element_type = T;

    KMM_HOST_DEVICE void consume(T x) {
        value = value > x ? value : x;
    }

    KMM_HOST_DEVICE T finish() {
        return value;
    }

    static constexpr T identity_value = std::numeric_limits<T>::lowest();
    T value = identity_value;
};

// `BitwiseAnd`/`BitwiseOr` are defined for integer `T` only.

template<typename T>
struct Reducer<T, ReductionOp::BitwiseAnd, std::enable_if_t<std::is_integral_v<T>>> {
    using element_type = T;

    KMM_HOST_DEVICE void consume(T x) {
        value &= x;
    }

    KMM_HOST_DEVICE T finish() {
        return value;
    }

    T value = static_cast<T>(~static_cast<T>(0));
};

template<typename T>
struct Reducer<T, ReductionOp::BitwiseOr, std::enable_if_t<std::is_integral_v<T>>> {
    using element_type = T;

    KMM_HOST_DEVICE void consume(T x) {
        value |= x;
    }

    KMM_HOST_DEVICE T finish() {
        return value;
    }

    T value = static_cast<T>(0);
};

// `KeyValue<VT>` (argmin/argmax): ordered by value with the key as tie-breaker, so only `Min`/`Max`
// are defined. The identity carries the neutral value (the largest/smallest `VT`) and key `0`.

template<typename VT>
struct Reducer<KeyValue<VT>, ReductionOp::Min> {
    using element_type = KeyValue<VT>;

    KMM_HOST_DEVICE void consume(KeyValue<VT> x) {
        value = value < x ? value : x;
    }

    KMM_HOST_DEVICE KeyValue<VT> finish() {
        return value;
    }

    static constexpr VT identity_value = std::numeric_limits<VT>::max();
    KeyValue<VT> value = {0, identity_value};
};

template<typename VT>
struct Reducer<KeyValue<VT>, ReductionOp::Max> {
    using element_type = KeyValue<VT>;

    KMM_HOST_DEVICE void consume(KeyValue<VT> x) {
        value = value > x ? value : x;
    }

    KMM_HOST_DEVICE KeyValue<VT> finish() {
        return value;
    }

    static constexpr VT identity_value = std::numeric_limits<VT>::lowest();
    KeyValue<VT> value = {0, identity_value};
};

/// `true` when `Reducer<T, Op>` is a complete type, i.e. `Op` is a supported reduction for element
/// type `T`. Used by the dispatch to reject unsupported combinations (e.g. `BitwiseAnd` on `float`,
/// `Sum` on `KeyValue`) with a runtime error instead of a compile error.
template<typename T, ReductionOp Op, typename = void>
inline constexpr bool is_reduction_supported = false;

template<typename T, ReductionOp Op>
inline constexpr bool is_reduction_supported<T, Op, std::void_t<decltype(sizeof(Reducer<T, Op>))>> =
    true;

}  // namespace kmm::memops
