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

    /// Acquire a new partial buffer for the given reduction. The buffer holds `count *
    /// replication` elements, laid out as `count` contiguous groups of `replication` elements
    /// each (group `i` occupies elements `[i * replication, (i + 1) * replication)`). Each group
    /// is folded down to a single value -- the partial's actual contribution to output slot `i`
    /// -- before being combined with any other partial.
    MemoryBuffer acquire_partial(
        ReductionState& reduction,
        MemoryId memory_id,
        DeviceStream stream_hint
    );

    /// Transition the reduction from open (i.e., partial buffers can still be acquired) to
    /// active (i.e., the reduction will be performed). Use `is_done` to check when the reduction
    /// is actually completed.
    void submit_reduction(ReductionState& reduction, MemoryId memory_id);

    /// return `true` if the reduction has been finalized, i.e., acquire_partial can no longer
    /// be called to acquire partial buffers.
    bool is_submitted(const ReductionState& reduction);

    Poll poll_reduction(ReductionState& reduction);

    void release_reduction(ReductionState& reduction);

  private:
    MemoryManager& m_memory_manager;
    refcnt_ptr<MemorySystem> m_memory_system;
};

}  // namespace kmm
