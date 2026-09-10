#pragma once

#include <vector>

#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memops/types.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/utils/poll.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

class MemorySystem;
class ReductionStateImpl;
using ReductionState = refcnt_ptr<ReductionStateImpl>;

KMM_REFCNT_TRAITS_FWD(ReductionStateImpl)

class ReductionJob;

class ReductionManager {
  public:
    ReductionManager(MemoryManager& memory_manager, refcnt_ptr<MemorySystem> memory_system);
    ~ReductionManager();

    /// Initialize a new reduction for the given buffer.
    ReductionState initialize_reduction(
        MemoryBuffer home_buffer,
        ReductionOp op,
        DataType dtype,
        size_t count
    );

    /// Acquire a new partial buffer for the given reduction.
    MemoryBuffer acquire_partial(
        ReductionState& reduction,
        MemoryId memory_id,
        const DeviceStreamId& stream_hint
    );

    /// Check that a contribution combining with `op` over `dtype` elements is compatible with the
    /// parameters this reduction was opened with by `begin_reduction`. Throws `std::runtime_error`
    /// on a mismatch, since the partials are folded using the reduction's own op/dtype and a
    /// mismatch would silently produce a wrong result.
    void check_compatible(const ReductionState& reduction, ReductionOp op, DataType dtype) const;

    /// Transition the reduction from open (i.e., partial buffers can still be acquired) to
    /// active (i.e., the reduction will be performed).
    void submit_reduction(ReductionState& reduction, MemoryId memory_id);

    bool is_submitted(const ReductionState& reduction);

    Poll poll_reduction(ReductionState& reduction);

    void release_reduction(ReductionState& reduction);

  private:
    MemoryManager& m_memory_manager;
    refcnt_ptr<MemorySystem> m_memory_system;
};

}  // namespace kmm
