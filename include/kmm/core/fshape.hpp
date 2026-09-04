#pragma once

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/domain_traits.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/range.hpp"
#include "kmm/core/type_utils.hpp"
#include "kmm/core/vec.hpp"

namespace kmm {

/// \addtogroup geometry
/// @{

/// The extent (size) of an N-dimensional domain along each axis, backed by a `Vec<T, N>`, using
/// Fortran-style 1-based indexing: axis `i` covers the valid indices `[1, extent]` instead of the
/// C-style `[0, extent)` used by `Shape`.
///
/// `FShape` is structurally identical to `Shape` (same storage and extent semantics, i.e.
/// `extent(i)` still just means "how many elements along axis `i`"); the indexing convention only
/// takes effect through the `detail::domain_traits<FShape<N, T>>` specialization used by `Layout`.
template<size_t N, typename T = default_index_type>
class FShape: public Vec<T, N> {
  public:
    using storage_type = Vec<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr FShape(const storage_type& storage) : storage_type(storage) {}

    /// Create an empty shape `(0, 0, ...)`
    KMM_HOST_DEVICE
    constexpr FShape() : storage_type(::kmm::fill<N>(static_cast<T>(0))) {}

    constexpr FShape(const FShape&) = default;
    constexpr FShape(FShape&&) noexcept = default;
    FShape& operator=(const FShape&) = default;
    FShape& operator=(FShape&&) noexcept = default;

    /// Create a shape `(first, args...)`
    template<typename... Ts, typename = enable_if_t<N == 1 + sizeof...(Ts)>>
    KMM_HOST_DEVICE FShape(T first, Ts&&... args) : storage_type {first, args...} {}

    /// Create a shape from another shape. Throws on overflow.
    template<size_t M, typename U>
    KMM_HOST_DEVICE constexpr FShape(const FShape<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = FShape::from(that);
    }

    /// Create a shape from another shape. Does not throw on overflow.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr FShape from(const Vec<U, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? static_cast<T>(that[i]) : static_cast<T>(1);
        }

        return FShape(result);
    }

    /// Create a shape `(value, value, value, ...)`.
    KMM_HOST_DEVICE
    static constexpr FShape fill(T value) {
        return FShape(::kmm::fill<N>(value));
    }

    /// Create a shape `(1, 1, 1, ...)`.
    KMM_HOST_DEVICE
    static constexpr FShape one() {
        return fill(static_cast<T>(1));
    }

    /// Create a shape `(0, 0, 0, ...)`.
    KMM_HOST_DEVICE
    static constexpr FShape zero() {
        return fill(static_cast<T>(0));
    }

    /// Returns coordinate i, or default_value if the axis is out of range.
    KMM_HOST_DEVICE
    T get_or_default(size_t i, T default_value = static_cast<T>(1)) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(is_less(i, N))) {
                return (*this)[i];
            }
        }

        return default_value;
    }

    /// Checks whether this shape can be converted to `FShape<M, U>`.
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

    /// Check if this shape is empty.
    ///
    /// A shape is empty if each axis is less than or equal to zero.
    KMM_HOST_DEVICE
    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= !(static_cast<T>(0) < (*this)[i]);
        }

        return result;
    }

    /// Returns the product of the extents of this shape.
    ///
    /// If one of the axis is negative, the returned value is zero.
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

    /// Check if a point falls within this shape, using 1-based indices.
    ///
    /// This means that for each axis `p[i] >= 1` and `p[i] <= shape[i]`.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE bool contains(const Point<M, U>& p) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N) && is_less(i, M); i++) {
            result &= !is_less(p[i], static_cast<U>(1)) && !is_less((*this)[i], p[i]);
        }

        if constexpr (N < M) {
            for (size_t i = N; is_less(i, M); i++) {
                result &= is_equal(p[i], static_cast<U>(1));
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
FShape(Ts&&...) -> FShape<sizeof...(Ts)>;

/// Constructs an FShape from the given per-axis extents.
template<typename... Ts>
KMM_HOST_DEVICE FShape<sizeof...(Ts)> fshape(const Ts&... values) {
    return FShape<sizeof...(Ts)> {
        Vec<default_index_type, sizeof...(Ts)> {static_cast<default_index_type>(values)...}
    };
}

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE FShape<N + M, T> concat(const FShape<N, T>& lhs, const FShape<M, T>& rhs) {
    return FShape<N + M, T> {
        concat(static_cast<const Vec<T, N>&>(lhs), static_cast<const Vec<T, M>&>(rhs))
    };
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator==(const FShape<N, T>& lhs, const FShape<M, U>& rhs) {
    bool result = true;

    for (size_t i = 0; is_less(i, N) || is_less(i, M); i++) {
        result &= is_equal(lhs.get_or_default(i), rhs.get_or_default(i));
    }

    return result;
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator!=(const FShape<N, T>& lhs, const FShape<M, U>& rhs) {
    return !(lhs == rhs);
}

/// @}

namespace detail {

// Same extent semantics as `domain_traits<Shape<N, IndexT>>` -- only `bounds()` differs, since
// `FShape` is Fortran-style 1-based: valid indices along an axis are `[1, extent]` rather than
// `[0, extent)`. `extent`/`slice_axis`/`drop_axis`/`permute_axes`/`insert_axis` all operate on
// extents (counts), which are indexing-convention-agnostic, so their logic is identical to
// `Shape`'s, just retargeted to produce `FShape` instead of `Shape`.
template<size_t N, typename IndexT>
struct domain_traits<FShape<N, IndexT>> {
    static constexpr size_t rank = N;
    using index_type = IndexT;
    using domain_type = FShape<N, index_type>;

    KMM_HOST_DEVICE
    static constexpr Range<index_type> bounds(const domain_type& domain, size_t axis) {
        return {static_cast<index_type>(1), domain[axis] + static_cast<index_type>(1)};
    }

    KMM_HOST_DEVICE
    static constexpr index_type extent(const domain_type& domain, size_t axis) {
        return domain[axis];
    }

    template<size_t Axis>
    using slice_axis_type = FShape<N, index_type>;

    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr slice_axis_type<Axis> slice_axis(
        domain_type domain,
        index_type begin,
        index_type end
    ) {
        domain[Axis] = end - begin;
        return domain;
    }
    template<size_t Axis>
    using drop_axis_type = FShape<N - 1, index_type>;

    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr drop_axis_type<Axis> drop_axis(const domain_type& domain) {
        return permute_axes(domain, drop_index_sequence<rank, Axis>());
    }

    /// Runtime-axis counterpart of `drop_axis<Axis>`. The result type only depends on `N` (not on
    /// which axis is dropped), so `axis` does not need to be known at compile time here.
    KMM_HOST_DEVICE
    static constexpr drop_axis_type<0> drop_axis(const domain_type& domain, size_t axis) {
        KMM_ASSERT(axis < N);
        drop_axis_type<0> result;
        size_t j = 0;

        for (size_t i = 0; is_less(i, N); i++) {
            if (i != axis) {
                result[j] = domain[i];
                j++;
            }
        }

        return result;
    }

    template<size_t... Is>
    using permute_axes_type = FShape<sizeof...(Is), index_type>;

    template<size_t... Is>
    KMM_HOST_DEVICE static constexpr permute_axes_type<Is...> permute_axes(
        const domain_type& domain,
        IndexSequence<Is...>
    ) {
        return {domain[Is]...};
    }

    template<size_t Axis>
    using insert_axis_type = FShape<N + 1, index_type>;

    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr insert_axis_type<Axis> insert_axis(
        const domain_type& domain,
        index_type extent
    ) {
        insert_axis_type<Axis> result;

        for (size_t i = 0; is_less(i, Axis); i++) {
            result[i] = domain[i];
        }

        result[Axis] = extent;

        for (size_t i = Axis; is_less(i, N); i++) {
            result[i + 1] = domain[i];
        }

        return result;
    }
};

}  // namespace detail

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct fmt::formatter<kmm::FShape<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::FShape<N, T>>: std::hash<kmm::Vec<T, N>> {};
#endif
