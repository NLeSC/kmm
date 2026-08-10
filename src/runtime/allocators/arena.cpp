#include <algorithm>
#include <utility>

#include "kmm/runtime/allocators/arena.hpp"
#include "kmm/core/panic.hpp"

namespace kmm {

static constexpr size_t ARENA_ALIGNMENT = 256;

static size_t round_up_to_alignment(size_t n) {
    return (n + ARENA_ALIGNMENT - 1) / ARENA_ALIGNMENT * ARENA_ALIGNMENT;
}

ArenaAllocator::ArenaAllocator(std::unique_ptr<AsyncAllocator> inner, size_t block_size) :
    m_inner(std::move(inner)),
    m_block_size(block_size) {}

ArenaAllocator::~ArenaAllocator() {
    poll();

    for (auto& block : m_blocks) {
        m_inner->deallocate_async(DeviceStream{}, block->base, BufferLayout {block->size}, {});
    }
}

void ArenaAllocator::insert_free(Block& block, size_t offset, size_t size, DeviceEventSet deps) {
    block.by_size.insert({size, offset});
    block.by_offset.emplace(offset, Chunk {size, std::move(deps)});
}

ArenaAllocator::Chunk ArenaAllocator::take_free(Block& block, size_t offset) {
    auto it = block.by_offset.find(offset);
    KMM_ASSERT(it != block.by_offset.end());

    Chunk chunk = std::move(it->second);
    block.by_offset.erase(it);
    block.by_size.erase({chunk.size, offset});

    return chunk;
}

bool ArenaAllocator::find_best_fit(
    const DeviceStream& stream,
    size_t nbytes,
    Block*& block_out,
    size_t& offset_out
) const {
    // Case 1: prefer a free chunk whose last use is already guaranteed to precede `stream`,
    // so reusing it needs no extra wait. This is common when the same stream is reused.
    size_t num_blocks = m_blocks.size();

    for (size_t k = 0; k < num_blocks; k++) {
        size_t block_index = (m_search_hint + k) % num_blocks;
        const Block& block = *m_blocks[block_index];

        for (auto it = block.by_size.lower_bound({nbytes, 0}); it != block.by_size.end(); ++it) {
            if (stream.preceded_by(block.by_offset.at(it->second).deps)) {
                block_out = m_blocks[block_index].get();
                offset_out = it->second;
                m_search_hint = block_index;
                return true;
            }
        }
    }

    // Case 2: no ready chunk exists anywhere, so fall back to the smallest fitting chunk.
    for (size_t k = 0; k < num_blocks; k++) {
        size_t block_index = (m_search_hint + k) % num_blocks;
        auto it = m_blocks[block_index]->by_size.lower_bound({nbytes, 0});

        if (it == m_blocks[block_index]->by_size.end()) {
            continue;
        }

        block_out = m_blocks[block_index].get();
        offset_out = it->second;
        m_search_hint = block_index;
        return true;
    }

    return false;
}

AllocResult ArenaAllocator::add_block(const DeviceStream& stream, size_t min_size) {
    min_size = std::max(min_size, size_t(1024));
    size_t size = std::max(m_block_size, min_size);
    void* base;

    while (true) {
        DeviceEventSet deps;
        AllocResult result = m_inner->allocate_async(stream, BufferLayout {size}, &base, deps);

        if (result == AllocResult::Success) {
            auto block = std::make_unique<Block>();
            block->base = base;
            block->size = size;
            insert_free(*block, 0, size, std::move(deps));
            m_blocks.push_back(std::move(block));
            return AllocResult::Success;
        }

        // if we drop below the request size, we give up.
        if (size < min_size) {
            return result;
        }
    }
}

AllocResult ArenaAllocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out,
    DeviceEventSet& deps_out
) {
    size_t nbytes = round_up_to_alignment(layout.size_in_bytes);

    Block* block;
    size_t offset;

    if (!find_best_fit(stream, nbytes, block, offset)) {
        AllocResult result = add_block(stream, nbytes);

        if (result != AllocResult::Success) {
            return result;
        }

        offset = 0;
        block = m_blocks.back().get();
    }

    Chunk chunk = take_free(*block, offset);

    if (chunk.size > nbytes) {
        insert_free(*block, offset + nbytes, chunk.size - nbytes, chunk.deps);
    }

    deps_out.insert(std::move(chunk.deps));
    *addr_out = static_cast<char*>(block->base) + offset;
    m_allocations[*addr_out] = {block, offset, nbytes};
    return AllocResult::Success;
}

void ArenaAllocator::deallocate_async(
    const DeviceStream& stream,
    void* addr, BufferLayout layout, DeviceEventSet deps) {
    auto it = m_allocations.find(addr);
    KMM_ASSERT(it != m_allocations.end());

    Allocation alloc = it->second;
    m_allocations.erase(it);

    Block& block = *alloc.block;
    size_t offset = alloc.offset;
    size_t size = alloc.size;

    // Coalesce with the preceding free chunk
    auto succ_it = block.by_offset.lower_bound(offset);
    if (succ_it != block.by_offset.begin()) {
        auto pred_it = std::prev(succ_it);

        if (pred_it->first + pred_it->second.size == offset) {
            size_t pred_offset = pred_it->first;
            Chunk pred_chunk = take_free(block, pred_offset);

            offset = pred_offset;
            size += pred_chunk.size;
            deps.insert(std::move(pred_chunk.deps));
        }
    }

    // Coalesce with the next free chunk
    auto next_it = block.by_offset.find(offset + size);
    if (next_it != block.by_offset.end()) {
        size_t next_offset = next_it->first;
        Chunk next_chunk = take_free(block, next_offset);

        size += next_chunk.size;
        deps.insert(std::move(next_chunk.deps));
    }

    insert_free(block, offset, size, std::move(deps));
}

void ArenaAllocator::poll() {
    m_inner->poll();
}

void ArenaAllocator::trim(size_t nbytes_remaining) {
    size_t total_bytes = 0;
    for (auto& block : m_blocks) {
        total_bytes += block->size;
    }

    for (size_t i = 0; i < m_blocks.size() && total_bytes > nbytes_remaining;) {
        Block& block = *m_blocks[i];
        bool fully_free = block.by_offset.size() == 1 && block.by_offset.begin()->first == 0
            && block.by_offset.begin()->second.size == block.size;

        if (!fully_free) {
            i++;
            continue;
        }

        for (const auto& event : block.by_offset.begin()->second.deps) {
            event.synchronize();
        }

        m_inner->deallocate_async(DeviceStream{}, block.base, BufferLayout {block.size}, {});
        total_bytes -= block.size;
        m_blocks.erase(m_blocks.begin() + i);
    }

    m_inner->trim(nbytes_remaining);
}

}  // namespace kmm
