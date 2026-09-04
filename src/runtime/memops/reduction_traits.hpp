#pragma once

#include <limits>
#include <type_traits>

#include "kmm/core/key_value.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/runtime/memops/types.hpp"

namespace kmm::memops {

/// Per-`ReductionOp` identity element and combining function, specialized per `T`/`Op` so that the
/// CPU (`reduce`) and GPU (`reduce_gpu`) reduction kernels can fold elements without a runtime
/// branch or function-pointer indirection to pick the operator. `KMM_HOST_DEVICE` so the same
/// definition serves both.
///
/// The primary template is left undefined: a `ReductionTraits<T, Op>` is a complete type exactly
/// when the `(T, Op)` pair is a supported reduction. `is_reduction_supported<T, Op>` derives from
/// that, and the dispatch in `reduction.cpp` / `reduction_gpu.cu` guards on it before instantiating
/// a kernel.
template<typename T, ReductionOp Op, typename = void>
struct ReductionTraits;

template<typename T>
struct ReductionTraits<T, ReductionOp::Sum, std::enable_if_t<std::is_arithmetic_v<T>>> {
    KMM_HOST_DEVICE static T identity() {
        return static_cast<T>(0);
    }

    KMM_HOST_DEVICE static T combine(T a, T b) {
        return static_cast<T>(a + b);
    }
};

template<typename T>
struct ReductionTraits<T, ReductionOp::Product, std::enable_if_t<std::is_arithmetic_v<T>>> {
    KMM_HOST_DEVICE static T identity() {
        return static_cast<T>(1);
    }

    KMM_HOST_DEVICE static T combine(T a, T b) {
        return static_cast<T>(a * b);
    }
};

template<typename T>
struct ReductionTraits<T, ReductionOp::Min, std::enable_if_t<std::is_arithmetic_v<T>>> {
    KMM_HOST_DEVICE static T identity() {
        return std::numeric_limits<T>::max();
    }

    KMM_HOST_DEVICE static T combine(T a, T b) {
        return a < b ? a : b;
    }
};

template<typename T>
struct ReductionTraits<T, ReductionOp::Max, std::enable_if_t<std::is_arithmetic_v<T>>> {
    KMM_HOST_DEVICE static T identity() {
        return std::numeric_limits<T>::lowest();
    }

    KMM_HOST_DEVICE static T combine(T a, T b) {
        return a > b ? a : b;
    }
};

// `BitwiseAnd`/`BitwiseOr` are defined for integer `T` only.

template<typename T>
struct ReductionTraits<T, ReductionOp::BitwiseAnd, std::enable_if_t<std::is_integral_v<T>>> {
    KMM_HOST_DEVICE static T identity() {
        return static_cast<T>(~static_cast<T>(0));
    }

    KMM_HOST_DEVICE static T combine(T a, T b) {
        return static_cast<T>(a & b);
    }
};

template<typename T>
struct ReductionTraits<T, ReductionOp::BitwiseOr, std::enable_if_t<std::is_integral_v<T>>> {
    KMM_HOST_DEVICE static T identity() {
        return static_cast<T>(0);
    }

    KMM_HOST_DEVICE static T combine(T a, T b) {
        return static_cast<T>(a | b);
    }
};

// `KeyValue<VT>` (argmin/argmax): ordered by value with the key as tie-breaker, so only `Min`/`Max`
// are defined. The identity carries the neutral value (the largest/smallest `VT`) and key `0`.

template<typename VT>
struct ReductionTraits<KeyValue<VT>, ReductionOp::Min> {
    KMM_HOST_DEVICE static KeyValue<VT> identity() {
        return KeyValue<VT>(0, std::numeric_limits<VT>::max());
    }

    KMM_HOST_DEVICE static KeyValue<VT> combine(KeyValue<VT> a, KeyValue<VT> b) {
        return a < b ? a : b;
    }
};

template<typename VT>
struct ReductionTraits<KeyValue<VT>, ReductionOp::Max> {
    KMM_HOST_DEVICE static KeyValue<VT> identity() {
        return KeyValue<VT>(0, std::numeric_limits<VT>::lowest());
    }

    KMM_HOST_DEVICE static KeyValue<VT> combine(KeyValue<VT> a, KeyValue<VT> b) {
        return a > b ? a : b;
    }
};

/// `true` when `ReductionTraits<T, Op>` is a complete type, i.e. `Op` is a supported reduction for
/// element type `T`. Used by the dispatch to reject unsupported combinations (e.g. `BitwiseAnd` on
/// `float`, `Sum` on `KeyValue`) with a runtime error instead of a compile error.
template<typename T, ReductionOp Op, typename = void>
inline constexpr bool is_reduction_supported = false;

template<typename T, ReductionOp Op>
inline constexpr bool
    is_reduction_supported<T, Op, std::void_t<decltype(sizeof(ReductionTraits<T, Op>))>> = true;

}  // namespace kmm::memops
