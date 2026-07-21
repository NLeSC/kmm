#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/domain.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

template<size_t N>
struct ArrayChunk {
    MemoryId owner_id;
    Point<N> offset;
    Dim<N> size;
};

template<size_t N>
class Distribution {
  public:
    Distribution();
    Distribution(Dim<N> array_size, Dim<N> chunk_size, std::vector<MemoryId> memories);

    static Distribution from_chunks(
        Dim<N> array_size,
        std::vector<ArrayChunk<N>> chunks,
        bool allow_duplicates = false
    );

    size_t region_to_chunk_index(Bounds<N> region) const;

    ArrayChunk<N> chunk(size_t index) const;

    size_t num_chunks() const {
        return m_memories.size();
    }

    Dim<N> chunk_size() const {
        return m_chunk_size;
    }

    Dim<N> array_size() const {
        return m_array_size;
    }

  protected:
    Dim<N> m_array_size = Dim<N>::zero();
    Dim<N> m_chunk_size = Dim<N>::zero();
    std::array<size_t, N> m_chunks_count;
    std::vector<MemoryId> m_memories;
};

template<size_t N, typename M>
Distribution<N> map_domain_to_distribution(
    Dim<N> array_size,
    const Domain& domain,
    M mapper,
    bool allow_duplicates = false
) {
    std::vector<ArrayChunk<N>> chunks;

    for (const auto& chunk : domain.chunks) {
        Bounds<N> bounds = mapper(chunk, Bounds<N>(array_size));

        chunks.push_back(ArrayChunk<N> {
            .owner_id = chunk.owner_id.as_memory(),
            .offset = bounds.begin(),
            .size = bounds.size()
        });
    }

    return Distribution<N>::from_chunks(array_size, chunks, allow_duplicates);
}

#define KMM_INSTANTIATE_ARRAY_IMPL(NAME) \
    template class NAME<0>; /* NOLINT */ \
    template class NAME<1>; /* NOLINT */ \
    template class NAME<2>; /* NOLINT */ \
    template class NAME<3>; /* NOLINT */ \
    template class NAME<4>; /* NOLINT */ \
    template class NAME<5>; /* NOLINT */ \
    template class NAME<6>; /* NOLINT */

}  // namespace kmm