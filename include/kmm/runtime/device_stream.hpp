#pragma once

#include <initializer_list>
#include <ostream>

#include "kmm/core/panic.hpp"
#include "kmm/utils/function_ref.hpp"
#include "kmm/utils/gpu_utils.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/refcnt_ptr.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class DeviceStream;
class DeviceEvent;
class DeviceEventSet;

// Free-standing (not nested) so `KMM_REFCNT_TRAITS_FWD` can be declared before `DeviceStream`,
// which needs it available already for the inline `is_null`/`operator<`/`operator==` below.
class DeviceStreamImpl;
KMM_REFCNT_TRAITS_FWD(DeviceStreamImpl)

class DeviceStream {
  public:
    using index_type = uint8_t;

    DeviceStream() noexcept = default;
    static DeviceStream create(CUcontext context, CUstream stream, bool destroy_if_done);

    DeviceEvent record_event() const;
    bool is_ready() const noexcept;
    void make_progress() const;

    void attach_callback(NotifyHandle callback) const;

    // Native handle access
    cuda_stream_id id() const;
    CUstream get() const;
    CUcontext context() const;

    void wait_on_default_stream() const;
    void wait_on_event(CUevent event) const;
    void wait_on_event(const DeviceEvent& event) const;
    void wait_on_events(const DeviceEventSet& events) const;

    void synchronize() const;

    // Ordering queries: is `src` already guaranteed to happen before `dst`.
    // Note: this is a hint. If `true` then `src` MUST precede `dst. However, if
    // it returns `false`, then `src` MAY still precede `dst`.
    bool preceded_by(const DeviceEvent& src) const;
    bool preceded_by(const DeviceEventSet& src) const;

    // True if some event in `deps` was recorded on this stream and nothing has been
    // recorded on this stream since, i.e. reusing this stream for the next operation
    // would not force it to queue behind unrelated work.
    bool is_latest_in(const DeviceEventSet& deps) const;

    void attach_callback(uint64_t event_id, NotifyHandle callback) const;
    void synchronize(uint64_t event_id) const;
    bool is_ready(uint64_t event_id) const;
    bool is_latest(uint64_t event_id) const;

    DeviceEvent with_stream(const DeviceEventSet& pred, function_ref<void(CUstream)> fun) const;

    bool with_event(uint64_t event_id, function_ref<void(CUstream, CUevent)> callback) const;

    bool is_null() const {
        return m_impl == nullptr;
    }

    friend bool operator<(const DeviceStream& a, const DeviceStream& b) {
        return a.m_impl.get() < b.m_impl.get();
    }

    friend bool operator==(const DeviceStream& a, const DeviceStream& b) {
        return a.m_impl.get() == b.m_impl.get();
    }

    using Impl = DeviceStreamImpl;

  private:
    friend class DeviceStreamImpl;

    explicit DeviceStream(refcnt_ptr<Impl> impl) noexcept;

    refcnt_ptr<Impl> m_impl;
};

}  // namespace kmm

template<>
struct fmt::formatter<kmm::DeviceStream>: fmt::ostream_formatter {};