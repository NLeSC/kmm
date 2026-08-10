#pragma once

#include <cstddef>
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
};

class Requisition {
    KMM_NOT_COPYABLE_OR_MOVABLE(Requisition)

  public:
    explicit Requisition(MemoryId memory_id, MemoryTransaction parent = {});
    ~Requisition();

    /// Registers `buffer_id` for access with the given mode. Only valid while `Building`. Returns
    /// the index of this buffer's entry, valid for `accessor(index)` once `Ready`.
    size_t add(BufferId buffer_id, AccessMode mode);

    /// Registers `buffer_id` for reduce-mode access. The buffer must already be in reduction
    /// mode (see `Runtime::begin_reduction`). Only valid while `Building`. Returns the index of
    /// this buffer's entry, valid for `accessor(index)` once `Ready`.
    size_t add_reduction(BufferId buffer_id);

    /// The accessor for the buffer added at `index` (as returned by `add`). Only valid once
    /// `Ready`.
    BufferAccessor accessor(Runtime& runtime, size_t index) const;

    const DeviceStream& stream() const noexcept {
        KMM_ASSERT(m_state == RequisitionState::Ready);
        return m_stream;
    }

    /// Events that must complete before it is safe to read/write the accessors above. Only valid
    /// once `Ready`.
    const DeviceEventSet& dependencies() const noexcept {
        KMM_ASSERT(m_state == RequisitionState::Ready);
        return m_deps;
    }

  private:
    friend class Runtime;

    MemoryId m_memory_id;
    DeviceStream m_stream;
    MemoryTransaction m_parent;
    std::vector<RequisitionDep> m_entries;
    DeviceEventSet m_deps;
    RequisitionState m_state = RequisitionState::Building;
};

}  // namespace kmm
