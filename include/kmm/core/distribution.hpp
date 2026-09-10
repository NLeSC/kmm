#pragma once

#include "kmm/core/bounds.hpp"
#include "kmm/core/checked_compare.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/core/point.hpp"
#include "kmm/core/shape.hpp"

namespace kmm {

/// \addtogroup geometry
/// @{

/// Describes how an N-dimensional `total_shape` is partitioned into a row-major grid of
/// chunks, each of (at most) `chunk_shape` elements along every axis.
template<size_t N, typename IndexT = default_index_type>
class Distribution {
  public:
    using index_type = IndexT;
    using shape_type = Shape<N, IndexT>;
    using point_type = Point<N, IndexT>;

    KMM_HOST_DEVICE
    constexpr Distribution() = default;

    KMM_HOST_DEVICE
    constexpr Distribution(shape_type total_shape, shape_type chunk_shape) :
        m_total_shape(total_shape),
        m_chunk_shape(chunk_shape) {
        shape_type grid_shape;

        for (size_t i = 0; is_less(i, N); i++) {
            auto extent = total_shape[i];
            auto chunk = chunk_shape[i];
            grid_shape[i] = chunk > static_cast<IndexT>(0)
                ? static_cast<IndexT>((extent + chunk - 1) / chunk)
                : static_cast<IndexT>(0);
        }

        m_grid_shape = grid_shape;
    }

    /// The full extent of the distributed domain.
    KMM_HOST_DEVICE
    shape_type total_shape() const noexcept {
        return m_total_shape;
    }

    /// The nominal (maximum) extent of a single chunk; chunks at the edge of the domain may be
    /// smaller, see `chunk_extent`.
    KMM_HOST_DEVICE
    shape_type chunk_shape() const noexcept {
        return m_chunk_shape;
    }

    /// The number of chunks along each axis.
    KMM_HOST_DEVICE
    shape_type grid_shape() const noexcept {
        return m_grid_shape;
    }

    /// The total number of chunks (`grid_shape().volume()`).
    KMM_HOST_DEVICE
    size_t num_chunks() const noexcept {
        return static_cast<size_t>(m_grid_shape.volume());
    }

    /// The actual extent of the chunk at the given grid index, clipped at the edges of
    /// `total_shape`.
    KMM_HOST_DEVICE
    shape_type chunk_extent(point_type grid_index) const noexcept {
        shape_type extent;

        for (size_t i = 0; is_less(i, N); i++) {
            auto begin = grid_index[i] * m_chunk_shape[i];
            auto end = begin + m_chunk_shape[i];

            if (end > m_total_shape[i]) {
                end = m_total_shape[i];
            }

            extent[i] = end > begin ? end - begin : static_cast<IndexT>(0);
        }

        return extent;
    }

    /// The global offset (origin) of the chunk at the given grid index.
    KMM_HOST_DEVICE
    point_type chunk_offset(point_type grid_index) const noexcept {
        point_type offset;

        for (size_t i = 0; is_less(i, N); i++) {
            offset[i] = grid_index[i] * m_chunk_shape[i];
        }

        return offset;
    }

    /// Row-major linearization of a grid index (the last axis varies fastest).
    KMM_HOST_DEVICE
    size_t linear_index(point_type grid_index) const noexcept {
        size_t linear = 0;

        for (size_t i = 0; is_less(i, N); i++) {
            linear =
                linear * static_cast<size_t>(m_grid_shape[i]) + static_cast<size_t>(grid_index[i]);
        }

        return linear;
    }

    /// The range of grid indices (per axis, half-open `[begin, end)`) of the chunks that overlap
    /// `region`. An axis with no overlap results in an empty range along that axis, which makes
    /// the returned `Bounds` empty as a whole (see `Bounds::is_empty`).
    KMM_HOST_DEVICE
    Bounds<N, IndexT> chunk_range(const Bounds<N, IndexT>& region) const noexcept {
        Bounds<N, IndexT> result;

        for (size_t i = 0; is_less(i, N); i++) {
            auto lo =
                region.begin(i) > static_cast<IndexT>(0) ? region.begin(i) : static_cast<IndexT>(0);
            auto hi = region.end(i) < m_total_shape[i] ? region.end(i) : m_total_shape[i];

            if (m_chunk_shape[i] > static_cast<IndexT>(0) && lo < hi) {
                auto begin_chunk = lo / m_chunk_shape[i];
                auto end_chunk =
                    (hi - static_cast<IndexT>(1)) / m_chunk_shape[i] + static_cast<IndexT>(1);

                if (end_chunk > m_grid_shape[i]) {
                    end_chunk = m_grid_shape[i];
                }

                result[i] = Range<IndexT> {begin_chunk, end_chunk};
            }
        }

        return result;
    }

    /// Inverse of `linear_index`: recovers the grid index of the chunk with the given row-major
    /// linear index.
    KMM_HOST_DEVICE
    point_type unravel(size_t linear) const noexcept {
        point_type grid_index;

        for (size_t i = N; i-- > 0;) {
            auto extent = static_cast<size_t>(m_grid_shape[i]);
            grid_index[i] = static_cast<IndexT>(extent > 0 ? linear % extent : 0);
            linear = extent > 0 ? linear / extent : 0;
        }

        return grid_index;
    }

  private:
    shape_type m_total_shape {};
    shape_type m_chunk_shape {};
    shape_type m_grid_shape {};
};

/// @}

}  // namespace kmm
