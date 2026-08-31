#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/buffer.hpp"
#include "kmm/runtime/data_interfaces/base.hpp"
#include "kmm/runtime/device_event.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/poll.hpp"
#include "kmm/utils/refcnt_ptr.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

// MemoryRequestImpl / MemoryRequest are forward-declared in buffer.hpp (included above), since
// BufferRequest needs them too and this header would otherwise have to be included before it.
class MemoryTransactionImpl;
class MemoryBufferImpl;
class MemoryRequestImpl;

KMM_REFCNT_TRAITS_FWD(MemoryTransactionImpl)
KMM_REFCNT_TRAITS_FWD(MemoryBufferImpl)
KMM_REFCNT_TRAITS_FWD(MemoryRequestImpl)

using MemoryBuffer = refcnt_ptr<MemoryBufferImpl>;
using MemoryTransaction = refcnt_ptr<MemoryTransactionImpl>;
using MemoryRequest = refcnt_ptr<MemoryRequestImpl>;

enum struct AccessKind { ReadOnly, SharedWrite, Exclusive };

class MemoryManager {
  public:
    struct Impl;

    MemoryManager();
    ~MemoryManager();

    MemoryBuffer create_buffer(
        std::unique_ptr<DataInterface> data,
        std::string name,
        bool evictable = true,
        std::optional<MemoryId> home_memory_id = {}
    );

    void release_buffer(MemoryBuffer buffer);

    /// Creates a new transaction, optionally as a child of `parent`.
    MemoryTransaction create_transaction(MemoryTransaction parent = {});

    MemoryRequest create_request(
        const MemoryBuffer& buffer,
        MemoryId memory_id,
        AccessKind mode,
        MemoryTransaction parent = {},
        NotifyHandle callback = {}
    );

    Poll poll_request(const DeviceStreamId& stream_hint, const MemoryRequest& request);

    /// Returns the accessor for a request that has reached `Ready` (see `poll_request`).
    BufferAccessor access_request(const MemoryRequest& request, DeviceEventSet& deps_out);

    void release_request(MemoryRequest request, const DeviceEventSet& deps = {});

    void prefetch_buffer(
        const MemoryBuffer& buffer,
        MemoryId memory_id,
        AccessKind mode = AccessKind::ReadOnly
    );

    void try_evict_buffer(const MemoryBuffer& buffer, MemoryId memory_id);

    void invalidate_buffer(const MemoryBuffer& buffer);

    void trim_device(DeviceId id, size_t bytes_remaining = 0);

    void make_progress();

  private:
    std::unique_ptr<Impl> m_impl;
};

std::ostream& operator<<(std::ostream& stream, AccessKind access);

}  // namespace kmm

template<>
struct fmt::formatter<kmm::AccessKind>: fmt::ostream_formatter {};