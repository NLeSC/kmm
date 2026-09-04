#pragma once

#include <cstddef>
#include <exception>
#include <vector>

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

struct BufferRequest {
    MemoryId memory_id;
    BufferId buffer_id;
    AccessMode mode;
};

class ResourceRequest {
  public:
    size_t add(MemoryId memory_id, BufferId buffer_id, AccessMode mode) {
        size_t index = m_requests.size();
        m_requests.push_back({memory_id, buffer_id, mode});
        return index;
    }

  private:
    friend class Runtime;

    std::vector<BufferRequest> m_requests;
};

class ResourceGrant {
    KMM_NOT_COPYABLE_OR_MOVABLE(ResourceGrant)

  public:
    BufferAccessor accessor(size_t index) const noexcept {
        return m_entries.at(index).accessor;
    }

    const DeviceEventSet& dependencies() const noexcept {
        return m_deps;
    }

    const MemoryTransaction& transaction() const noexcept {
        return m_transaction;
    }

  private:
    friend class Runtime;

    struct Entry {
        BufferId buffer_id;
        MemoryRequest request;
        BufferAccessor accessor {};
    };

    ResourceGrant(std::vector<Entry> entries, DeviceEventSet deps, MemoryTransaction transaction) :
        m_entries(std::move(entries)),
        m_deps(std::move(deps)),
        m_transaction(std::move(transaction)) {}

    std::vector<Entry> m_entries;
    DeviceEventSet m_deps;
    MemoryTransaction m_transaction;
};

}  // namespace kmm
