#pragma once

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/core/vec.hpp"

#if !KMM_IS_RTC
    #include <stdexcept>
#endif

namespace kmm {

/// \addtogroup utility
/// @{

/// A precomputed divisor that replaces integer division by a cheaper multiply-and-shift.
///
/// Uses the "round-up" magic-number scheme of T. Granlund and P. L. Montgomery,
/// "Division by Invariant Integers using Multiplication", ACM SIGPLAN PLDI 1994,
/// pp. 61-72 (https://doi.org/10.1145/178243.178249): the quotient is
/// `(numerator + mulhi(numerator, M)) >> shift`, where `M` is the low 32 (resp. 64)
/// bits of `ceil(2^(N+shift) / divisor)` and the always-set bit `N` is folded back in
/// as `+ numerator`.
template<typename T>
class FastDivisor;

template<>
class FastDivisor<uint32_t> {
  public:
    /// The largest numerator that `divide` and `modulo` accept.
    static constexpr uint32_t max_numerator = (uint32_t(1) << 31) - 1;

    /// Construct a divisor equal to one (the identity).
    constexpr FastDivisor() = default;

    /// Precompute the magic constants for dividing by `divisor`.
    ///
    /// Throws `std::runtime_error` (or panics on device) when `divisor` is zero.
    KMM_HOST_DEVICE
    explicit constexpr FastDivisor(uint32_t divisor) : m_divisor(divisor) {
        if (divisor == 0) {
#if KMM_IS_DEVICE
            KMM_PANIC("FastDivisor: divisor must be non-zero");
#elif !KMM_IS_RTC
            throw std::runtime_error("FastDivisor: divisor must be non-zero");
#endif
        }

        // shift = ceil(log2(divisor))
        uint32_t shift = 0;
        while ((uint64_t(1) << shift) < divisor) {
            shift++;
        }

        if (shift >= 32) {
            // The divisor exceeds every admissible numerator, so the quotient is always 0.
            m_multiplier = 0;
            m_shift = 31;
        } else {
            // The magic number M = ceil(2^(32 + shift) / divisor) needs 33 bits; bit 32 is
            // always set, so only the low 32 bits are stored and `divide` adds the numerator
            // back to account for the implicit high bit. For a power of two M == 2^32
            // exactly, hence m_multiplier == 0 and `divide` reduces to a plain shift.
            const uint64_t t = (uint64_t(1) << shift) - divisor;  // 2^shift - divisor
            m_multiplier = static_cast<uint32_t>(((t << 32) + divisor - 1) / divisor);
            m_shift = shift;
        }
    }

    /// Return `numerator / divisor`. Requires `numerator <= max_numerator`.
    KMM_HOST_DEVICE
    constexpr uint32_t divide(uint32_t numerator) const {
        // KMM_ASSERT(numerator <= max_numerator);
#if KMM_IS_DEVICE
        // `mulhi(a, b)` == high 32 bits of the 32x32 -> 64 bit product.
        const uint32_t high = __umulhi(numerator, m_multiplier);
#else
        const uint32_t high = static_cast<uint32_t>((uint64_t(numerator) * m_multiplier) >> 32);
#endif
        return (numerator + high) >> m_shift;
    }

    /// Return `numerator % divisor`. Requires `numerator <= max_numerator`.
    KMM_HOST_DEVICE
    constexpr uint32_t modulo(uint32_t numerator) const {
        return numerator - divide(numerator) * m_divisor;
    }

    /// Return the divisor.
    KMM_HOST_DEVICE
    constexpr uint32_t get() const {
        return m_divisor;
    }

  private:
    uint32_t m_divisor = 1;
    uint32_t m_multiplier = 0;  // low bits of the magic number; 0 for a power of two
    uint32_t m_shift = 0;
};

template<>
class FastDivisor<uint64_t> {
  public:
    /// The largest numerator that `divide` and `modulo` accept.
    static constexpr uint64_t max_numerator = (uint64_t(1) << 63) - 1;

    /// Construct a divisor equal to one (the identity).
    constexpr FastDivisor() = default;

    /// Precompute the magic constants for dividing by `divisor`.
    ///
    /// Throws `std::runtime_error` (or panics on device) when `divisor` is zero.
    KMM_HOST_DEVICE
    explicit constexpr FastDivisor(uint64_t divisor) : m_divisor(divisor) {
        if (divisor == 0) {
#if KMM_IS_DEVICE
            KMM_PANIC("FastDivisor: divisor must be non-zero");
#elif !KMM_IS_RTC
            throw std::runtime_error("FastDivisor: divisor must be non-zero");
#endif
        }

        // shift = ceil(log2(divisor))
        uint32_t shift = 0;
        while (((unsigned __int128)(1) << shift) < divisor) {
            shift++;
        }

        if (shift >= 64) {
            // The divisor exceeds every numerator, so division always returns zero.
            m_multiplier = 0;
            m_shift = 63;
        } else {
            // The magic number M = ceil(2^(64 + shift) / divisor) needs 65 bits; bit 64 is
            // always set, so only the low 64 bits are stored and `divide` adds the numerator
            // back to account for the implicit high bit. For a power of two M == 2^64
            // exactly, hence m_multiplier == 0 and `divide` reduces to a plain shift.
            const unsigned __int128 t = ((unsigned __int128)(1) << shift) - divisor;
            m_multiplier = static_cast<uint64_t>(((t << 64) + divisor - 1) / divisor);
            m_shift = shift;
        }
    }

    /// Return `numerator / divisor`. Requires `numerator <= max_numerator`.
    KMM_HOST_DEVICE
    constexpr uint64_t divide(uint64_t numerator) const {
        // KMM_ASSERT(numerator <= max_numerator);
#if KMM_IS_DEVICE
        // `mulhi(a, b)` == high 64 bits of the 64x64 -> 128 bit product.
        const uint64_t high = __umul64hi(numerator, m_multiplier);
#else
        const uint64_t high =
            static_cast<uint64_t>(((unsigned __int128)(numerator)*m_multiplier) >> 64);
#endif
        return (numerator + high) >> m_shift;
    }

    /// Return `numerator % divisor`. Requires `numerator <= max_numerator`.
    KMM_HOST_DEVICE
    constexpr uint64_t modulo(uint64_t numerator) const {
        return numerator - divide(numerator) * m_divisor;
    }

    /// Return the divisor.
    KMM_HOST_DEVICE
    constexpr uint64_t get() const {
        return m_divisor;
    }

  private:
    uint64_t m_divisor = 1;
    uint64_t m_multiplier = 0;  // low bits of the magic number; 0 for a power of two
    uint32_t m_shift = 0;
};

template<typename T>
KMM_HOST_DEVICE constexpr T operator/(T numerator, const FastDivisor<T>& divisor) {
    return divisor.divide(numerator);
}

template<typename T>
KMM_HOST_DEVICE constexpr T operator%(T numerator, const FastDivisor<T>& divisor) {
    return divisor.modulo(numerator);
}

/// Converts between a flat index and its `N`-dimensional coordinates and back.
///
/// The conversion is `flat_index == coord[0] + extent[0] * (coord[1] + extent[1] * (... + extent[N-2] * coord[N-1]))`.
/// This means extent[0] is the most-contiguous index.
template<typename T, size_t N>
class IndexMapper {
  public:
    /// The largest `volume()` that `unravel` supports.
    static constexpr T max_volume = FastDivisor<T>::max_numerator + T(1);

    IndexMapper() = default;

    KMM_HOST_DEVICE
    explicit IndexMapper(const T* extents) {
        for (size_t i = 0; i < N - 1; i++) {
            m_extents[i] = FastDivisor<T>(extents[i]);
        }

        m_volume = checked_product(extents, extents + N);
    }

    KMM_HOST_DEVICE
    explicit IndexMapper(const Vec<T, N>& extents) : IndexMapper(&extents[0]) {}

    KMM_HOST_DEVICE
    T volume() const noexcept {
        return m_volume;
    }

    /// Decompose `index` into `result`. Returns `true` if `index` lies within the space.
    ///
    /// `result` is fully overwritten, so the caller need not initialize it.
    KMM_HOST_DEVICE
    bool unravel(T linear_index, Vec<T, N>& result) const {
        if (linear_index >= m_volume) {
            return false;
        }

        for (size_t i = 0; i < N - 1; i++) {
            const T quo = linear_index / m_extents[i];
            result[i] = linear_index - quo * m_extents[i].get();
            linear_index = quo;
        }

        result[N - 1] = linear_index;
        return true;
    }

    /// Combine the coordinates in `coord` back into a flat index.
    KMM_HOST_DEVICE
    T ravel(const Vec<T, N>& coord) const {
        T linear_index = coord[N - 1];
        for (size_t i = N - 1; i-- > 0;) {
            linear_index = linear_index * m_extents[i].get() + coord[i];
        }
        return linear_index;
    }

  private:
    FastDivisor<T> m_extents[N - 1];  // we do not need store the last extent
    T m_volume = 1;
};

template<typename T>
class IndexMapper<T, 1> {
  public:
    /// A 1-D mapping performs no division, so the volume is limited only by `T`.
    static constexpr T max_volume = ~T(0);

    IndexMapper() = default;

    KMM_HOST_DEVICE
    explicit IndexMapper(const T* extents) {
        m_extent = extents[0];
    }

    KMM_HOST_DEVICE
    explicit IndexMapper(const Vec<T, 1>& extents) : IndexMapper(&extents.x) {}

    KMM_HOST_DEVICE
    T volume() const noexcept {
        return m_extent;
    }

    /// Decompose `index` into `result`. Returns `true` if `index` lies within the space.
    ///
    /// `result` is fully overwritten, so the caller need not initialize it.
    KMM_HOST_DEVICE
    bool unravel(T linear_index, Vec<T, 1>& result) const {
        result[0] = linear_index;
        return linear_index < m_extent;
    }

    /// Combine the coordinates in `coord` back into a flat index.
    KMM_HOST_DEVICE
    T ravel(const Vec<T, 1>& coord) const {
        return coord[0];
    }

  private:
    T m_extent = 1;
};

template<typename T>
class IndexMapper<T, 0> {
  public:
    /// A 0-D mapping performs no division, so the volume is limited only by `T`.
    static constexpr T max_volume = ~T(0);

    IndexMapper() = default;

    KMM_HOST_DEVICE
    explicit IndexMapper(const T*) {}

    KMM_HOST_DEVICE
    explicit IndexMapper(const Vec<T, 0>& extents) {}

    KMM_HOST_DEVICE
    T volume() const noexcept {
        return 0;
    }

    KMM_HOST_DEVICE
    bool unravel(T index, Vec<T, 0>&) const {
        return index == 0;
    }

    KMM_HOST_DEVICE
    T ravel(const Vec<T, 0>&) const {
        return 0;
    }
};

/// @}

}  // namespace kmm
