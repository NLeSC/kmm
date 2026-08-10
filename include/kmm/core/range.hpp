#pragma once

#include "kmm/core/checked_compare.hpp"

namespace kmm {

/// \addtogroup geometry
/// @{

/// Represents a half-open range of numbers beginning at `start` and ending at `stop`, excluding
/// the `stop` value itself. For example, `Range<int>(5, 10)` represents the numbers
/// `5, 6, 7, 8, 9`, but does not include `10`.
///
/// A range with `start < stop` is considered valid and non-empty (size is `stop-start`).
/// A range with `start == stop` is considered valid but empty (size is 0).
/// A range with `start > stop` is considered invalid (most operations treat this as empty).
///
/// Ranges can be iterated over directly:
///
/// ```
/// for (auto i : Range<int>(5, 10)) {
///   printf("%d\n", i); // prints 5, 6, 7, 8, 9
/// }
/// ```
template<typename T>
class Range {
  public:
    struct iterator;
    using value_type = T;
    using const_iterator = iterator;

    constexpr Range(const Range&) = default;
    constexpr Range(Range&&) noexcept = default;

    constexpr Range& operator=(const Range&) = default;
    constexpr Range& operator=(Range&&) noexcept = default;

    /// Constructs an empty range `0...0`
    KMM_HOST_DEVICE
    constexpr Range() = default;

    /// Constructs the range `0...stop`
    KMM_HOST_DEVICE
    explicit constexpr Range(T stop) : stop(stop) {}

    /// Constructs the range `start...stop`
    KMM_HOST_DEVICE
    constexpr Range(T start, T stop) : start(start), stop(stop) {}

    /// Converts a range from another range. Throws exception on overflow.
    template<typename U = T>
    KMM_HOST_DEVICE constexpr Range(const Range<U>& that) {
        if (!that.template is_convertible_to<T>()) {
            throw_overflow_exception();
        }

        *this = Range::from(that);
    }

    /// Converts a range from another range. Does not check for overflow.
    template<typename U = T>
    KMM_HOST_DEVICE static constexpr Range from(const Range<U>& range) {
        return {static_cast<T>(range.start), static_cast<T>(range.stop)};
    }

    /// Returns the range 0...1
    KMM_HOST_DEVICE static constexpr Range one() {
        return {static_cast<T>(0), static_cast<T>(1)};
    }

    /// Returns whether both bounds can be converted to `U` without overflow.
    template<typename U>
    KMM_HOST_DEVICE constexpr bool is_convertible_to() const {
        return is_convertible<U>(start) && is_convertible<U>(stop);
    }

    /// Checks if the range is empty (i.e., `begin == end`) or invalid (i.e., `begin > end`).
    KMM_HOST_DEVICE
    constexpr bool is_empty() const noexcept {
        return !(this->start < this->stop);
    }

    /// Returns an iterator to the first element.
    KMM_HOST_DEVICE
    constexpr iterator begin() const noexcept {
        return this->start;
    }

    /// Returns an iterator to one passed the last element.
    KMM_HOST_DEVICE
    constexpr iterator end() const noexcept {
        return is_empty() ? this->start : this->stop;
    }

    // Checks if the given index `index` is within this range.
    template<typename U = T>
    KMM_HOST_DEVICE constexpr bool contains(const U& index) const noexcept {
        return is_less_equal(this->start, index) & is_less(index, this->stop);
    }

    /// Returns whether `that` is fully contained within this range.
    /// Empty ranges are considered contained. Invalid ranges are never contained.
    template<typename U = T>
    KMM_HOST_DEVICE constexpr bool contains(const Range<U>& that) const noexcept {
        return is_less_equal(this->start, that.start) &  //
            is_less_equal(that.stop, this->stop) &  //
            is_less_equal(this->start, this->stop) &  //
            is_less_equal(that.start, that.stop);
    }

    /// Returns whether this range and `that` overlap with non-empty intersection.
    /// Empty or invalid ranges never overlap.
    template<typename U = T>
    KMM_HOST_DEVICE constexpr bool overlaps(const Range<U>& that) const noexcept {
        return is_less(this->start, this->stop) &  //
            is_less(that.start, that.stop) &  //
            is_less(this->start, that.stop) &  //
            is_less(that.start, this->stop);
    }

    /// Returns the range that lies in the intersection of `this` and `that`.
    KMM_HOST_DEVICE
    constexpr Range intersection(const Range& that) const noexcept {
        return {
            this->start > that.start ? this->start : that.start,
            this->stop < that.stop ? this->stop : that.stop,
        };
    }

    /// Computes the size (or length) of the range. This always a non-negative number as
    /// empty or invalid ranges will have a length of zero.
    KMM_HOST_DEVICE
    constexpr T size() const noexcept {
        return this->start >= this->stop ? static_cast<T>(0) : this->stop - this->start;
    }

    struct Pair {
        Range first;
        Range second;
    };

    /// Returns the ranges `start..mid` and `mid..stop`. Example
    ///
    /// ```
    /// auto [left, right] = some_range.split(10);
    /// ```
    KMM_HOST_DEVICE
    constexpr Pair split(T mid) const {
        if (mid < this->start) {
            mid = this->start;
        }

        if (mid > this->stop) {
            mid = this->stop;
        }

        return {{start, mid}, {mid, stop}};
    }

    /// The start point of the range.
    T start = static_cast<T>(0);

    /// The end point of the range (not inclusive).
    T stop = static_cast<T>(0);
};

template<typename T>
Range(const T&) -> Range<T>;

template<typename T>
Range(const T&, const T&) -> Range<T>;

/// Constructs the range `0...stop`.
template<typename T>
KMM_HOST_DEVICE constexpr Range<T> range(const T& stop) {
    return Range<T>(static_cast<T>(0), stop);
}

/// Constructs the range `start...stop`.
template<typename T>
KMM_HOST_DEVICE constexpr Range<T> range(const T& start, const T& stop) {
    return Range<T>(start, stop);
}

template<typename L, typename R = L>
KMM_HOST_DEVICE constexpr bool operator==(const Range<L>& lhs, const Range<R>& rhs) {
    return is_equal(lhs.start, rhs.start) && is_equal(lhs.stop, rhs.stop);
}

template<typename L, typename R = L>
KMM_HOST_DEVICE constexpr bool operator!=(const Range<L>& lhs, const Range<R>& rhs) {
    return !(lhs == rhs);
}

/// Returns `Range(range.start + offset, range.stop + offset)`
template<typename T>
KMM_HOST_DEVICE constexpr Range<T> operator+(const Range<T>& range, const T& offset) {
    return {range.start + offset, range.stop + offset};
}

/// Returns `Range(range.start + offset, range.stop + offset)`
template<typename T>
KMM_HOST_DEVICE constexpr Range<T> operator+(const T& offset, const Range<T>& range) {
    return {offset + range.start, offset + range.stop};
}

/// Returns `Range(range.start - offset, range.stop - offset)`
template<typename T>
KMM_HOST_DEVICE constexpr Range<T> operator-(const Range<T>& range, const T& offset) {
    return {range.start - offset, range.stop - offset};
}

template<typename T>
class Range<T>::iterator {
  public:
    using value_type = T;
    using difference_type =
        decltype(static_cast<T*>(nullptr) - static_cast<T*>(nullptr));  //std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    KMM_HOST_DEVICE
    constexpr iterator(T value) : current(value) {}

    KMM_HOST_DEVICE
    constexpr const T& get() const {
        return this->current;
    }

    KMM_HOST_DEVICE
    constexpr T& get() {
        return this->current;
    }

    KMM_HOST_DEVICE
    constexpr operator T() const {
        return this->current;
    }

    KMM_HOST_DEVICE
    constexpr T operator*() const {
        return this->current;
    }

    KMM_HOST_DEVICE
    constexpr iterator& operator++() {
        ++this->current;
        return *this;
    }

    KMM_HOST_DEVICE
    constexpr iterator operator++(int) {
        return iterator(this->current++);
    }

    KMM_HOST_DEVICE
    friend constexpr bool operator==(const iterator& lhs, const iterator& rhs) {
        return lhs.current == rhs.current;
    }

    KMM_HOST_DEVICE
    friend constexpr bool operator!=(const iterator& lhs, const iterator& rhs) {
        return lhs.current != rhs.current;
    }

    T current;
};

/// @}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>
    #include <utility>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {

template<typename T>
std::ostream& operator<<(std::ostream& stream, const Range<T>& p) {
    return stream << p.start << "..." << p.stop;
}

}  // namespace kmm

template<typename T>
struct fmt::formatter<kmm::Range<T>>: fmt::ostream_formatter {};

template<typename T>
struct std::hash<kmm::Range<T>> {
    size_t operator()(const kmm::Range<T>& p) const {
        return ::kmm::hash_fields(p.start, p.stop);
    }
};
#endif