#pragma once

#include "kmm/core/macros.hpp"

namespace kmm {

/// \addtogroup utility
/// @{

/// Pair of a value (type `T`) and an associated key (type `int64_t`).
///
/// The main purpose is that `KeyValue` pairs implement the `<` operator, allowed them to be sorted.
/// The pairs are ordered by value, with ties resolved by the key. This is useful to sort a list
/// of `KeyValue` pairs by value and still have the associated key.
template<typename ValueT>
struct alignas(2 * alignof(long)) KeyValue {
    using key_type = long;
    using value_type = ValueT;

    KMM_HOST_DEVICE
    constexpr KeyValue() = default;

    KMM_HOST_DEVICE
    constexpr KeyValue(key_type k, value_type v) : value(v), key(k) {}

    value_type value;
    key_type key;
};

template<typename T>
KMM_HOST_DEVICE bool operator==(const KeyValue<T>& a, const KeyValue<T>& b) {
    return (a.key == b.key) && (a.value == b.value);
}

template<typename T>
KMM_HOST_DEVICE bool operator<(const KeyValue<T>& a, const KeyValue<T>& b) {
    return (a.value < b.value) | ((a.value == b.value) & (a.key < b.key));
}

template<typename T>
KMM_HOST_DEVICE bool operator<=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return (a.value < b.value) | ((a.value == b.value) & (a.key <= b.key));
}

template<typename T>
KMM_HOST_DEVICE bool operator!=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return !(a == b);
}

template<typename T>
KMM_HOST_DEVICE bool operator>(const KeyValue<T>& a, const KeyValue<T>& b) {
    return b < a;
}

template<typename T>
KMM_HOST_DEVICE bool operator>=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return b <= a;
}

/// @}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {
template<typename T>
std::ostream& operator<<(std::ostream& stream, const KeyValue<T>& kv) {
    return stream << "{key=" << kv.key << ", value=" << kv.value << "}";
}
}  // namespace kmm

template<typename T>
struct fmt::formatter<kmm::KeyValue<T>>: fmt::ostream_formatter {};

template<typename T>
struct std::hash<kmm::KeyValue<T>> {
    size_t operator()(const kmm::KeyValue<T>& kv) const {
        return kmm::hash_fields(kv.key, kv.value);
    }
};
#endif