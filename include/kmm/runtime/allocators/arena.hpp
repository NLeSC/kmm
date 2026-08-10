#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class ArenaAllocator: public AsyncAllocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArenaAllocator)

    struct Chunk {
        size_t size;
        DeviceEventSet deps;
    };

    struct Block {
        void* base;
        size_t size;

        // Free chunks of this block. Indexed by offset (for coalescing with other free chunks)
        // and by (size, offset) (for finding available space).
        std::map<size_t, Chunk> by_offset;
        std::set<std::pair<size_t, size_t>> by_size;
    };

    struct Allocation {
        Block* block;
        size_t offset;
        size_t size;
    };

  public:
    ArenaAllocator(std::unique_ptr<AsyncAllocator> inner, size_t block_size = 512UL << 20);
    ~ArenaAllocator();

    AllocResult allocate_async(
        const DeviceStream& stream,
        BufferLayout layout,
        void** addr_out,
        DeviceEventSet& deps_out
    ) override final;

    void deallocate_async(
        const DeviceStream& stream,
        void* addr,
        BufferLayout layout,
        DeviceEventSet deps
    ) override final;

    void poll() override final;
    void trim(size_t nbytes_remaining) override final;

  private:
    AllocResult add_block(const DeviceStream& stream, size_t min_size);

    bool find_best_fit(
        const DeviceStream& stream,
        size_t nbytes,
        Block*& block_out,
        size_t& offset_out
    ) const;

    static void insert_free(Block& block, size_t offset, size_t size, DeviceEventSet deps);
    static Chunk take_free(Block& block, size_t offset);

    std::unique_ptr<AsyncAllocator> m_inner;
    size_t m_block_size;
    std::vector<std::unique_ptr<Block>> m_blocks;
    std::unordered_map<void*, Allocation> m_allocations;

    // Index of the block where the previous search succeeded. Just a search-order
    // hint (taken modulo `m_blocks.size()`), so it stays harmless across trims.
    mutable size_t m_search_hint = 0;
};

}  // namespace kmm
