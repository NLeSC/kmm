#pragma once

#include <cstddef>
#include <exception>
#include <vector>

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/runtime.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

class Runtime;

enum struct RequisitionState { Building, Submitted, Ready, Released };

struct RequisitionDep {
    MemoryId memory_id;
    BufferId buffer_id;
    AccessMode mode;
    MemoryRequest request = nullptr;
    BufferAccessor accessor {};
};

class Requisition {
  public:
    Requisition(const Requisition&) = delete;
    Requisition& operator=(const Requisition&) = delete;

    /// Moves `that`'s state into this object, leaving `that` in the `Released` state (so its
    /// destructor is a no-op). Requisition owns no external resources and nothing else holds a
    /// pointer into it, so this is a plain member-wise move plus a sentinel reset.
    Requisition(Requisition&& that) noexcept;
    Requisition& operator=(Requisition&& that) noexcept;

    explicit Requisition();
    ~Requisition();

    /// Registers `buffer_id` for access with the given mode. Only valid while `Building`. Returns
    /// the index of this buffer's entry, valid for `accessor(index)` once `Ready`.
    size_t add(MemoryId memory_id, BufferId buffer_id, AccessMode mode);

    /// Registers `buffer_id` for reduce-mode access. The buffer must already be in reduction
    /// mode (see `Runtime::begin_reduction`). Only valid while `Building`. Returns the index of
    /// this buffer's entry, valid for `accessor(index)` once `Ready`.
    size_t add_reduction(MemoryId memory_id, BufferId buffer_id);

    /// The accessor for the buffer added at `index` (as returned by `add`). Only valid once
    /// `Ready`.
    BufferAccessor accessor(size_t index) const;

    /// Poisons every buffer accessed by this requisition, so future accesses to them rethrow
    /// `reason` instead of succeeding. See `Runtime::poison_buffer`.
    void poison(Runtime& runtime, std::exception_ptr reason) const noexcept;

    /// Events that must complete before it is safe to read/write the accessors above. Only valid
    /// once `Ready`.
    const DeviceEventSet& dependencies() const noexcept {
        KMM_ASSERT(m_state == RequisitionState::Ready);
        return m_deps;
    }

    /// The transaction created for this requisition's requests (a child of the `parent` passed to
    /// the constructor). Only valid once `Submitted` or `Ready`; useful as the `parent` for a
    /// transaction nested inside the work this requisition guards.
    const MemoryTransaction& transaction() const noexcept {
        KMM_ASSERT(m_state != RequisitionState::Building);
        return m_transaction;
    }

  private:
    friend class Runtime;

    RequisitionState m_state = RequisitionState::Building;
    MemoryTransaction m_transaction;
    std::vector<RequisitionDep> m_entries;
    DeviceEventSet m_deps;
};

}  // namespace kmm
