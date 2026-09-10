#pragma once

#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

#include "kmm/api/array.hpp"
#include "kmm/api/context.hpp"
#include "kmm/api/device.hpp"
#include "kmm/api/parallel_for.hpp"
#include "kmm/core/bounds.hpp"
#include "kmm/core/distribution.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/identifiers.hpp"

namespace kmm {

/// A large N-dimensional array stored as a set of smaller chunks, each chunk being an
/// independent `Array<T, N, PolicyT>` (own buffer) with its own home `MemoryId`.
template<typename T, size_t N, typename PolicyT = RowMajor>
class DistArray {
  public:
    using element_type = T;
    using policy_type = PolicyT;
    static constexpr size_t rank = N;
    using shape_type = Shape<N>;
    using point_type = Point<N>;
    using bounds_type = Bounds<N>;
    using chunk_type = Array<T, N, PolicyT>;
    using slice_type = typename chunk_type::move_origin_type;

    DistArray() = default;

    /// Allocates one chunk (its own `Buffer`) per grid cell of `dist`. `home_fn(linear_index)`
    /// picks the home `MemoryId` of the chunk with the given row-major linear chunk index; if
    /// omitted, chunks are round-robined over the available GPUs
    /// (`runtime.system_info().num_devices()`), falling back to the host if there are none.
    DistArray(
        Runtime runtime,
        Distribution<N> dist,
        PolicyT policy = {},
        std::optional<T> fill_value = std::nullopt,
        std::function<MemoryId(size_t)> home_fn = {}
    ) :
        m_dist(dist) {
        if (!home_fn) {
            size_t num_devices = runtime.system_info().num_devices();
            home_fn = [num_devices](size_t linear_index) {
                return num_devices > 0 ? MemoryId::device(DeviceId(linear_index % num_devices))
                                       : MemoryId::host();
            };
        }

        m_chunks.reserve(dist.num_chunks());

        for (size_t i = 0; i < dist.num_chunks(); i++) {
            auto extent = dist.chunk_extent(dist.unravel(i));
            auto home = home_fn(i);

            m_chunks.push_back(
                ChunkEntry {chunk_type(runtime, extent, policy, fill_value, home), home}
            );
        }
    }

    /// The partitioning geometry (total shape, chunk shape, grid shape) of this array.
    const Distribution<N>& distribution() const noexcept {
        return m_dist;
    }

    /// The full (global) extent of this array.
    shape_type shape() const noexcept {
        return m_dist.total_shape();
    }

    /// The number of chunks along each axis.
    shape_type grid_shape() const noexcept {
        return m_dist.grid_shape();
    }

    /// The total number of chunks.
    size_t num_chunks() const noexcept {
        return m_chunks.size();
    }

    const chunk_type& chunk(size_t linear_index) const noexcept {
        return m_chunks[linear_index].array;
    }

    chunk_type& chunk(size_t linear_index) noexcept {
        return m_chunks[linear_index].array;
    }

    const chunk_type& chunk(point_type grid_index) const noexcept {
        return chunk(m_dist.linear_index(grid_index));
    }

    chunk_type& chunk(point_type grid_index) noexcept {
        return chunk(m_dist.linear_index(grid_index));
    }

    /// The home `MemoryId` of the chunk with the given row-major linear chunk index.
    MemoryId chunk_home(size_t linear_index) const noexcept {
        return m_chunks[linear_index].home;
    }

    /// The global offset (origin) of the chunk with the given row-major linear chunk index.
    point_type chunk_offset(size_t linear_index) const noexcept {
        return m_dist.chunk_offset(m_dist.unravel(linear_index));
    }

    /// Returns this array restricted to `region`, as a single sub-array (in global coordinates)
    /// of the one chunk that covers it. Panics if `region` does not overlap any chunk, or if it
    /// spans more than one chunk -- for a region that may cross chunk boundaries, use
    /// `distribution().chunk_range(region)` to enumerate the overlapping chunks yourself.
    slice_type slice(bounds_type region) const {
        auto grid_range = m_dist.chunk_range(region);

        if (grid_range.is_empty()) {
            KMM_PANIC("`slice` region does not overlap any chunk");
        }

        for (size_t axis = 0; axis < N; axis++) {
            if (grid_range.end(axis) - grid_range.begin(axis) != 1) {
                KMM_PANIC("`slice` region spans multiple chunks");
            }
        }

        auto linear_index = m_dist.linear_index(grid_range.begin());
        return m_chunks[linear_index].array.restrict_bounds(region);
    }

  private:
    struct ChunkEntry {
        chunk_type array;
        MemoryId home;
    };

    Distribution<N> m_dist;
    std::vector<ChunkEntry> m_chunks;
};

}  // namespace kmm
