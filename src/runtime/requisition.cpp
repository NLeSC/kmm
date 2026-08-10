#include <utility>

#include "kmm/runtime/requisition.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

Requisition::Requisition(MemoryId memory_id, MemoryTransaction parent) :
    m_memory_id(memory_id),
    m_parent(std::move(parent)) {}

Requisition::~Requisition() {
    KMM_ASSERT(m_state == RequisitionState::Released);
}

size_t Requisition::add(BufferId buffer_id, AccessMode mode) {
    KMM_ASSERT(m_state == RequisitionState::Building);

    size_t index = m_entries.size();
    m_entries.push_back({m_memory_id, buffer_id, mode});

    return index;
}

size_t Requisition::add_reduction(BufferId buffer_id) {
    KMM_ASSERT(m_state == RequisitionState::Building);

    size_t index = m_entries.size();
    m_entries.push_back({m_memory_id, buffer_id, AccessMode::Reduce});

    return index;
}

BufferAccessor Requisition::accessor(Runtime& runtime, size_t index) const {
    KMM_ASSERT(m_state == RequisitionState::Ready);
    return runtime.accessor(m_entries[index]);
}

}  // namespace kmm
