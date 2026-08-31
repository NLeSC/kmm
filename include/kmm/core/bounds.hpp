#pragma once

#include "kmm/core/domain_traits.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/range.hpp"
#include "kmm/core/shape.hpp"
#include "kmm/core/type_utils.hpp"
#include "kmm/core/vec.hpp"

namespace kmm {

/// \addtogroup geometry
/// @{

/// An N-dimensional axis-aligned box, given by a `Range<T>` per axis, backed by a
/// `Vec<Range<T>, N>`.
///
/// A `Bounds` describes a sub-region of an N-dimensional domain: the half-open `[begin, end)`
/// range of valid indices along each axis. It can be constructed from a begin/end point pair,
/// from an offset and a `Shape`, or directly from a `Shape` (assuming a zero origin). Use
/// `contains`/`overlaps`/`intersection` to test and combine bounds.
template<size_t N, typename T = default_index_type>
class Bounds: public Vec<Range<T>, N> {
  public:
    using storage_type = Vec<Range<T>, N>;

    KMM_HOST_DEVICE
    explicit constexpr Bounds(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Bounds() : storage_type(fill<N>(Range<T>())) {}

    constexpr Bounds(const Bounds&) = default;
    constexpr Bounds(Bounds&&) noexcept = default;
    Bounds& operator=(const Bounds&) = default;
    Bounds& operator=(Bounds&&) noexcept = default;

    template<typename... Ts, typename = enable_if_t<N == 1 + sizeof...(Ts)>>
    KMM_HOST_DEVICE Bounds(Range<T> first, Ts&&... args) : storage_type {first, args...} {}

    template<size_t M, typename U>
    KMM_HOST_DEVICE constexpr Bounds(const Bounds<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Bounds::from(that);
    }

    /// Construct from begin and end point.
    KMM_HOST_DEVICE static constexpr Bounds from_bounds(
        const Point<N, T>& begin,
        const Point<N, T>& end
    ) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = {begin[i], end[i]};
        }

        return Bounds(result);
    }

    /// Construct from offset and shape.
    KMM_HOST_DEVICE static constexpr Bounds from_offset_size(
        const Point<N, T>& offset,
        const Shape<N, T>& shape
    ) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = Range<T>(shape[i]) + offset[i];
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE
    constexpr Bounds(const Shape<N, T>& shape) :
        Bounds(from_offset_size(Point<N, T>::zero(), shape)) {}

    /// Returns an empty bounds (all ranges are 0...0).
    KMM_HOST_DEVICE static constexpr Bounds empty() {
        return Bounds(fill<N>(Range<T>()));
    }

    /// Returns `Bounds` with one element (all ranges are 0...1).
    KMM_HOST_DEVICE static constexpr Bounds one() {
        return Bounds(fill<N>(Range<T>::one()));
    }

    /// Returns `Bounds` constructed from another `Bounds`.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Bounds from(const Vec<Range<U>, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? Range<T>::from(that[i]) : Range<T>(static_cast<T>(1));
        }

        return Bounds(result);
    }

    /// Returns `true` if this `Bounds` can also be represented as `Bounds<M, U>` without
    /// loss of information, `false` otherwise.
    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (i < M) {
                result &= (*this)[i].template is_convertible_to<U>();
            } else {
                result &= (*this)[i] == Range<T>::one();
            }
        }

        return result;
    }

    /// Returns the range along the `i`-th dimension and `default_value` if it is out of bounds.
    KMM_HOST_DEVICE
    Range<T> get_or_default(size_t i, Range<T> default_value = Range<T>::one()) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(i < N)) {
                return (*this)[i];
            }
        }

        return default_value;
    }

    /// Returns the begin value along the `i`-th dimension.
    KMM_HOST_DEVICE
    T begin(size_t axis) const {
        return get_or_default(axis).start;
    }

    /// Returns the end value along the `i`-th dimension (exclusive).
    KMM_HOST_DEVICE
    T end(size_t axis) const {
        return get_or_default(axis).stop;
    }

    /// Returns the number of elements along the `i`-th dimension.
    KMM_HOST_DEVICE
    T size(size_t axis) const {
        return get_or_default(axis).size();
    }

    /// Returns N-d begin point.
    KMM_HOST_DEVICE
    Point<N, T> begin() const {
        Point<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = (*this)[axis].start;
        }
        return result;
    }

    /// Returns N-d end point (exclusive)
    KMM_HOST_DEVICE
    Point<N, T> end() const {
        Point<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = (*this)[axis].stop;
        }
        return result;
    }

    /// Returns the N-d shape (i.e., size along each dimension).
    KMM_HOST_DEVICE
    Shape<N, T> shape() const {
        Shape<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = (*this)[axis].size();
        }
        return result;
    }

    /// Returns `true` only if this bounds is empty.
    KMM_HOST_DEVICE
    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= this->begin(i) >= this->end(i);
        }

        return result;
    }

    /// Returns the product of `size(0) * size(1) * ...`.
    KMM_HOST_DEVICE
    T volume() const {
        T result = 1;

        for (size_t i = 0; is_less(i, N); i++) {
            result *= this->end(i) - this->begin(i);
        }

        return this->is_empty() ? T {0} : result;
    }

    KMM_HOST_DEVICE
    Bounds intersection(const Bounds& that) const {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i].start = this->begin(i) >= that.begin(i) ? this->begin(i) : that.begin(i);
            result[i].stop = this->end(i) <= that.end(i) ? this->end(i) : that.end(i);
        }

        return Bounds(result);
    }

    /// Returns `true` if this bounds overlaps the given bounds.
    KMM_HOST_DEVICE
    bool overlaps(const Bounds& that) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= this->begin(i) < this->end(i) && that.begin(i) < that.end(i) &&  //
                this->begin(i) < that.end(i) && that.begin(i) < this->end(i);
        }

        return result;
    }

    /// Returns `true` if this bounds contains the given bounds.
    KMM_HOST_DEVICE
    bool contains(const Bounds& that) const {
        bool is_contained = true;
        bool that_is_empty = false;

        for (size_t i = 0; is_less(i, N); i++) {
            is_contained &= that.begin(i) >= this->begin(i);
            is_contained &= that.end(i) <= this->end(i);
            that_is_empty |= that.begin(i) >= that.end(i);
        }

        return is_contained || that_is_empty;
    }

    /// Returns `true` if this bounds contains the given point.
    KMM_HOST_DEVICE
    bool contains(const Point<N, T>& that) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= (*this)[i].contains(that[i]);
        }

        return result;
    }

    /// Returns `true` if this bounds contains the given point `{first, rest, ...}`.
    //
    // NB: uses `enable_if_t<...>` directly rather than the `assert_arity_t` alias -- see the
    // comment on the analogous `Shape` constructor for why.
    template<typename... Ts, typename = enable_if_t<N == 1 + sizeof...(Ts)>>
    KMM_HOST_DEVICE bool contains(const T& first, Ts&&... rest) const {
        return contains(Point<N, T> {first, rest...});
    }

    /// Returns `true` if this bounds overlaps the given shape.
    KMM_HOST_DEVICE
    bool overlaps(const Shape<N, T>& that) const {
        return overlaps(Bounds<N, T> {that});
    }

    /// Returns `true` if this bounds contains the given shape.
    KMM_HOST_DEVICE
    bool contains(const Shape<N, T>& that) const {
        return contains(Bounds<N, T> {that});
    }
};

template<typename... Ts>
Bounds(Ts&&...) -> Bounds<sizeof...(Ts)>;

/// Constructs a `Bounds` from the given ranges.
template<typename... Ts>
KMM_HOST_DEVICE Bounds<sizeof...(Ts)> bounds(const Ts&... values) {
    return Bounds<sizeof...(Ts)> {Vec<Range<default_index_type>, sizeof...(Ts)> {values...}};
}

/// @}

namespace detail {

template<size_t N, typename IndexT>
struct domain_traits<Bounds<N, IndexT>> {
    static constexpr size_t rank = N;
    using index_type = IndexT;
    using domain_type = Bounds<N, index_type>;
    template<size_t Axis>
    using drop_axis_type = Bounds<N - 1, index_type>;

    KMM_HOST_DEVICE
    static constexpr Range<index_type> bounds(const domain_type& domain, size_t axis) {
        return domain[axis];
    }

    KMM_HOST_DEVICE
    static constexpr index_type extent(const domain_type& domain, size_t axis) {
        return domain[axis].size();
    }

    template<size_t Axis>
    using slice_axis_type = Bounds<N, index_type>;

    // Shifts the axis's range by `-begin` and truncates it to length `end - begin`, so the
    // returned domain is locally zero-based at this axis; the caller is responsible for
    // folding the absolute shift into a storage offset.
    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr slice_axis_type<Axis> slice_axis(
        domain_type domain,
        index_type begin,
        index_type end
    ) {
        domain[Axis] = Range<index_type> {end - begin};
        return domain;
    }

    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr drop_axis_type<Axis> drop_axis(const domain_type& domain) {
        return permute_axes(domain, drop_index_sequence<rank, Axis>());
    }

    template<size_t... Is>
    using permute_axes_type = Bounds<sizeof...(Is), index_type>;

    template<size_t... Is>
    KMM_HOST_DEVICE static constexpr permute_axes_type<Is...> permute_axes(
        const domain_type& domain,
        IndexSequence<Is...>
    ) {
        return {domain[Is]...};
    }

    template<size_t Axis>
    using insert_axis_type = Bounds<N + 1, index_type>;

    template<size_t Axis>
    KMM_HOST_DEVICE static constexpr insert_axis_type<Axis> insert_axis(
        const domain_type& domain,
        index_type extent
    ) {
        insert_axis_type<Axis> result;

        for (size_t i = 0; is_less(i, Axis); i++) {
            result[i] = domain[i];
        }

        result[Axis] = Range<index_type>(extent);

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
struct fmt::formatter<kmm::Bounds<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::Bounds<N, T>>: std::hash<kmm::Vec<kmm::Range<T>, N>> {};
#endif