#pragma once

#include "kmm/core/macros.hpp"

// This header is only ever meant to be pulled in from behind another header's own
// `#if !KMM_IS_RTC` guard (see vec.hpp/point.hpp/range.hpp), but it is self-guarded here too so
// that including it directly can never break NVRTC device compilation.
#if !KMM_IS_RTC
    #include <functional>
    #include <iterator>

namespace kmm {

/// \addtogroup utility
/// @{

template<typename T, typename = void>
struct Hasher: std::hash<T> {};

template<typename T, typename H = Hasher<T>>
size_t hash_value(const T& v, H hasher = {}) {
    return hasher(v);
}

template<typename T, typename H = Hasher<T>>
void hash_combine(size_t& seed, const T& v, H hasher = {}) {
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<typename It, typename H = Hasher<typename std::iterator_traits<It>::value_type>>
void hash_combine_range(size_t& seed, It begin, It end, H hasher = {}) {
    for (It current = begin; current != end; current++) {
        hash_combine(seed, *current, hasher);
    }
}

template<typename... Ts>
size_t hash_fields(const Ts&... fields) {
    size_t seed = 0;
    (hash_combine(seed, fields), ...);
    return seed;
}

template<typename It>
size_t hash_range(It begin, It end) {
    size_t seed = 0;
    hash_combine_range(seed, begin, end);
    return seed;
}

/// @}

}  // namespace kmm
#endif
