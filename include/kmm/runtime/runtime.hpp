#pragma once

#include <chrono>
#include <limits>
#include <optional>

#include "runtime_config.hpp"

#include "kmm/runtime/device_data_streams.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memops/types.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/runtime/system_info.hpp"
#include "kmm/utils/function_ref.hpp"

namespace kmm {

class RuntimeImpl;
class Requisition;

/// Top-level handle to the KMM runtime: owns the system's `SystemInfo`, `DeviceStreamRegistry`,
/// `MemorySystem`, and `MemoryManager`, and exposes the buffer/request operations built on top of
/// them. Cheap to copy (a `refcnt_ptr` to the shared `RuntimeImpl`).
class Runtime {
  public:
    /**
     * Returns the physical-memory backend used to allocate and move buffer data.
     */
    MemorySystem& memory_system() noexcept;

    /**
     * Returns the registry of device streams used to schedule and track work on the devices.
     */
    DeviceEventRegistry& event_registry() noexcept;

    /**
     * Returns information about the machine's topology (hosts, devices, memories).
     */
    const SystemInfo& system_info() const noexcept;

    /**
     * Poll the Runtime once. This will update bookkeeping that needs periodic updating and checks
     * if certain events have finished.
     *
     * @return The time when the next poll should happen. Polling before this time is a noop.
     */
    std::chrono::system_clock::time_point poll_once();

    /**
     * Repeatedly poll the runtime until the given callback return `true`. This is useful to wait
     * for a certain event to complete while the runtime system can still make progress in the
     * background.
     *
     * @param callback Called after each poll to check if we are done.
     * @param deadline Determines the maximum timeout.
     * @return Returns `true` if the callback return `true`, and `false` if the deadline expired.
     */
    bool poll_until_completion(
        function_ref<bool()> callback,
        std::chrono::system_clock::time_point deadline =
            std::chrono::system_clock::time_point::max()
    );

    /**
     * Create a new buffer in the runtime system.
     *
     * @param layout The layout of the new buffer.
     * @param name  The name of the new buffer.
     * @param fill_value If non-empty, the buffer's contents are set to repeated copies of this
     *  value the first time it is materialized in any memory.
     * @param home If set, the memory where the buffer is preferentially kept resident, used to
     *  pick a copy source once the buffer becomes valid there instead of the location of first
     *  access.
     * @return The identifier of the new buffer.
     */
    BufferId create_buffer(
        BufferLayout layout,
        std::string name,
        FillValue fill_value = {},
        std::optional<MemoryId> home = {}
    );

    /**
     * Release a buffer, freeing its memory once any pending accesses to it have completed.
     *
     * @param id The identifier of the buffer to release.
     */
    void release_buffer(BufferId id);

    /**
     * Puts a buffer into reduction mode: until `finalize_reduction` is called, the buffer may
     * only be accessed in reduce mode (`Requisition::add_reduction`). Any other access is a
     * bug and throws instead of being served.
     *
     * Throws if the buffer is already in reduction mode.
     */
    void begin_reduction(BufferId id, DataType dtype, ReductionOp op);

    /**
     * Finalizes the reduction previously started with `begin_reduction`, folding every value
     * written to the buffer during reduce-mode access into it and returning the buffer to
     * regular read/write access. The fold may still be pending when this call returns; `submit`
     * waits for it automatically before granting the next read/write access.
     *
     * Throws if the buffer is not currently in reduction mode.
     */
    void finalize_reduction(BufferId id, MemoryId memory_id = MemoryId::host());

    /**
     * Abandons the reduction previously started with `begin_reduction`: discards every value
     * written to the buffer during reduce-mode access (without folding any of it in) and returns
     * the buffer to regular read/write access. Unlike `finalize_reduction`, there is no fold to
     * wait for, so this cannot fail asynchronously. A no-op if the buffer is not currently in
     * reduction mode.
     */
    void rollback_reduction(BufferId id);

    /**
     * Ensure a valid copy of the buffer is available in the given memory, fetching it if needed.
     * This is just a hint. The runtime system might ignore the request (for example, if out of
     * memory or if the buffer is currently locked by another thread).
     */
    void prefetch_buffer(BufferId id, MemoryId memory_id, bool invalid_others = false);

    /**
     * Mark a buffer as poisoned, so future accesses to it rethrow `reason` instead of succeeding.
     * This is useful for cases where a write failed and the content of the buffer is now in an
     * invalid state, making it impossible for others to read.
     */
    void poison_buffer(BufferId id, std::exception_ptr reason) noexcept;

    /**
     * Returns a memory that currently holds a valid copy of the buffer, if any.
     */
    std::optional<MemoryId> find_valid_memory(BufferId) const;

    /**
     * Returns whether the given memory currently holds a valid copy of the buffer.
     */
    bool is_valid(BufferId id, MemoryId memory_id);

    /**
     * Returns whether the buffer currently has memory allocated for it in the given memory.
     */
    bool is_allocated(BufferId id, MemoryId memory_id);

    /**
     * Try to evict the buffer's copy from the given memory to free up space, if possible.
     */
    void try_evict_buffer(BufferId id, MemoryId memory_id);

    /**
     * Invalidate all copies of the buffer, discarding its contents.
     */
    void invalidate_buffer(BufferId id);

    /**
     * Submit a requisition. If a stream is provided, all required dependencies will be put onto
     * the stream and this method returns immediately. If no stream is provided, the method blocks
     * until the dependencies are available.
     */
    void submit(
        Requisition& req,
        std::optional<GPUStreamRef> stream = std::nullopt,
        MemoryTransaction parent = nullptr
    );

    /**
     * Release a submitted requisition, allowing others to access its buffers once `deps` complete.
     */
    void release(Requisition& req, DeviceEventSet deps = {});

    /**
     * Block the calling thread until the given device event has completed.
     */
    void synchronize(const DeviceEvent& e);

    /**
     * Block the calling thread until all device events in the set have completed.
     */
    void synchronize(const DeviceEventSet& e);

    /**
     * Block the calling thread until all events on all streams have completed.
     */
    void synchronize();

    /**
     * Copy from the given src buffer to the given dst buffer according to the given description.
     */
    DeviceEvent submit_copy(
        BufferId dst_id,
        BufferId src_id,
        CopyDescription description,
        MemoryId memory_id,
        MemoryTransaction parent = {}
    );

    explicit Runtime(refcnt_ptr<RuntimeImpl>);

  private:
    refcnt_ptr<RuntimeImpl> m_impl;
};

KMM_REFCNT_TRAITS_FWD(RuntimeImpl)

Runtime make_runtime(const RuntimeConfig& config = default_config_from_environment());

}  // namespace kmm