#include <utility>

#include "kmm/runtime/requisition.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

Requisition::Requisition() {}

Requisition::~Requisition() {
    KMM_ASSERT(m_state == RequisitionState::Released);
}

Requisition::Requisition(Requisition&& that) noexcept :
    m_state(std::exchange(that.m_state, RequisitionState::Released)),
    m_transaction(std::move(that.m_transaction)),
    m_entries(std::move(that.m_entries)),
    m_deps(std::move(that.m_deps)) {}

Requisition& Requisition::operator=(Requisition&& that) noexcept {
    if (this != &that) {
        KMM_ASSERT(m_state == RequisitionState::Released);
        m_state = std::exchange(that.m_state, RequisitionState::Released);
        m_transaction = std::move(that.m_transaction);
        m_entries = std::move(that.m_entries);
        m_deps = std::move(that.m_deps);
    }

    return *this;
}

size_t Requisition::add(MemoryId memory_id, BufferId buffer_id, AccessMode mode) {
    KMM_ASSERT(m_state == RequisitionState::Building);

    size_t index = m_entries.size();
    m_entries.push_back({memory_id, buffer_id, mode});

    return index;
}

size_t Requisition::add_reduction(MemoryId memory_id, BufferId buffer_id) {
    KMM_ASSERT(m_state == RequisitionState::Building);

    size_t index = m_entries.size();
    m_entries.push_back({memory_id, buffer_id, AccessMode::Reduce});

    return index;
}

BufferAccessor Requisition::accessor(size_t index) const {
    KMM_ASSERT(m_state == RequisitionState::Ready);
    return m_entries[index].accessor;
}

void Requisition::poison(Runtime& runtime, std::exception_ptr reason) const noexcept {
    for (const auto& entry : m_entries) {
        if (entry.mode != AccessMode::Read) {
            runtime.poison_buffer(entry.buffer_id, reason);
        }
    }
}

}  // namespace kmm
