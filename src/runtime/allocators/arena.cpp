#include <algorithm>
#include <utility>

#include "kmm/runtime/allocators/arena.hpp"
#include "kmm/core/panic.hpp"

namespace kmm {

static constexpr size_t ARENA_ALIGNMENT = 256;

static size_t round_up_to_alignment(size_t n) {
    return (n + ARENA_ALIGNMENT - 1) / ARENA_ALIGNMENT * ARENA_ALIGNMENT;
}

ArenaAllocator::ArenaAllocator(std::unique_ptr<Allocator> inner, size_t block_size) :
    m_inner(std::move(inner)),
    m_block_size(block_size) {}

ArenaAllocator::~ArenaAllocator() {
    poll();

    for (auto& block : m_blocks) {
        m_inner->deallocate(block->base, BufferLayout {block->size});
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
    const DeviceStream* stream_opt,
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
            const auto& deps = block.by_offset.at(it->second).deps;
            bool is_ready = stream_opt == nullptr ?
                                                  m_events.is_ready(deps) :
                                              stream_opt->preceded_by(deps);

            if (is_ready) {
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

AllocResult ArenaAllocator::add_block(const DeviceStream* stream_opt, size_t min_size) {
    min_size = std::max(min_size, size_t(1024));
    size_t size = std::max(m_block_size, min_size);
    void* base;

    while (true) {
        AllocResult result;
        DeviceEvent event {};

        if (stream_opt != nullptr) {
            result = m_inner->allocate_async(*stream_opt, BufferLayout {size}, &base);
        } else {
            result = m_inner->allocate(BufferLayout {size}, &base);
        }

        if (result == AllocResult::Success) {
            if (stream_opt != nullptr) {
                event = stream_opt->record_event();
            }

            auto block = std::make_unique<Block>();
            block->base = base;
            block->size = size;
            insert_free(*block, 0, size, event);
            m_blocks.push_back(std::move(block));
            m_bytes_reserved += size;
            return AllocResult::Success;
        }

        // if we drop below the request size, we give up.
        if (size <= min_size) {
            return result;
        }

        size = std::max(min_size, size / 2);
    }
}

AllocResult ArenaAllocator::allocate_generic(
    const DeviceStream* stream_opt,
    BufferLayout layout,
    void** addr_out
) {
    size_t nbytes = round_up_to_alignment(layout.size_in_bytes);

    Block* block;
    size_t offset;

    while (!find_best_fit(stream_opt, nbytes, block, offset)) {
        // did not find a block, try to add a block
        AllocResult result = add_block(stream_opt, nbytes);

        // new block added, we have our best fit
        if (result == AllocResult::Success) {
            offset = 0;
            block = m_blocks.back().get();
            break;
        }

        // Could not find a block, try to deallocate an empty block. There might be
        // enough memory available, just that the blocks are too fragmented.
        if (trim_one(stream_opt)) {
            continue;
        }

        // could not find a block, could not allocate a new block, could not free an unused block.
        // I am out of ideas. Just return that the allocation has failed.
        return result;
    }

    Chunk chunk = take_free(*block, offset);

    if (chunk.size > nbytes) {
        insert_free(*block, offset + nbytes, chunk.size - nbytes, chunk.deps);
    }

    if (stream_opt != nullptr) {
        stream_opt->wait_on_event(chunk.deps);
    } else {
        m_events.synchronize(chunk.deps);
    }

    *addr_out = static_cast<char*>(block->base) + offset;
    m_allocations[*addr_out] = {block, offset, nbytes};
    return AllocResult::Success;
}

void ArenaAllocator::deallocate_generic(
    const DeviceStream* stream_opt,
    void* addr, BufferLayout layout) {
    auto it = m_allocations.find(addr);
    KMM_ASSERT(it != m_allocations.end());

    Allocation alloc = it->second;
    m_allocations.erase(it);

    Block& block = *alloc.block;
    size_t offset = alloc.offset;
    size_t size = alloc.size;
    DeviceEventSet deps;

    if (stream_opt != nullptr) {
        deps.insert(stream_opt->record_event());
    }

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

AllocResult ArenaAllocator::allocate_async(
    const DeviceStream& stream,
    BufferLayout layout,
    void** addr_out
) {
    return allocate_generic(&stream, layout, addr_out);
}

void ArenaAllocator::deallocate_async(
    const DeviceStream& stream,
    void* addr,
    BufferLayout layout
) {
    deallocate_generic(&stream, addr, layout);
}

AllocResult ArenaAllocator::allocate(BufferLayout layout, void** addr_out) {
    return allocate_generic(nullptr, layout, addr_out);
}

void ArenaAllocator::deallocate(void* addr, BufferLayout layout) {
    deallocate_generic(nullptr, addr, layout);
}

void ArenaAllocator::poll() {
    m_inner->poll();
}

void ArenaAllocator::trim(size_t nbytes_remaining) {
    while (true) {
        // we can quit, enough bytes are available
        if (m_bytes_reserved <= nbytes_remaining) {
            break;
        }

        // try to release on unused block.
        if (!trim_one(nullptr)) {
            break;
        }
    }

    m_inner->trim(nbytes_remaining);
}

bool ArenaAllocator::trim_one(const DeviceStream* stream_opt) {
    for (size_t i = 0; i < m_blocks.size(); i++) {
        Block& block = *m_blocks[i];
        const auto& [offset, chunk] = *block.by_offset.begin();
        bool fully_free = block.by_offset.size() == 1 && offset == 0
            && chunk.size == block.size;

        if (fully_free) {
            const auto& event = chunk.deps;

            if (stream_opt != nullptr) {
                stream_opt->wait_on_event(event);
                m_inner->deallocate_async(*stream_opt, block.base, BufferLayout {block.size});
            } else {
                m_events.synchronize(event);
                m_inner->deallocate(block.base, BufferLayout {block.size});
            }

            m_bytes_reserved -= block.size;
            m_blocks.erase(m_blocks.begin() + i);
            return true;
        }
    }

    return false;
}

}  // namespace kmm
