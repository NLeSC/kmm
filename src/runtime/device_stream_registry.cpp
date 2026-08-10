#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <atomic>
#include <deque>
#include <mutex>
#include <queue>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/runtime/device_stream_registry.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

struct DeviceStreamRegistry::Impl {
    std::mutex mutex;
    ankerl::unordered_dense::map<cuda_stream_id, DeviceStream> streams;
};

DeviceStreamRegistry::DeviceStreamRegistry() : m_impl(std::make_unique<Impl>()) {}

DeviceStreamRegistry::~DeviceStreamRegistry() = default;

DeviceStream DeviceStreamRegistry::lookup(CUstream stream) {
    std::lock_guard guard {m_impl->mutex};
    auto it = m_impl->streams.find(stream);

    if (it != m_impl->streams.end()) {
        return it->second;
    }

    CUcontext context;
    KMM_CUDA_CHECK(cuStreamGetCtx(stream, &context));

    auto inserted = m_impl->streams.emplace(
        cuda_stream_id(stream),
        DeviceStream::create(context, stream, /*destroy_if_done=*/false)
    );
    return inserted.first->second;
}

DeviceStream DeviceStreamRegistry::create(CUcontext context, bool high_priority) {
    std::lock_guard guard {m_impl->mutex};

    CUstream stream;
    unsigned int flags = CU_STREAM_NON_BLOCKING;
    int least_priority;
    int greatest_priority;

    {
        CUDAContextGuard cguard {context};
        KMM_CUDA_CHECK(cuCtxGetStreamPriorityRange(&least_priority, &greatest_priority));

        int priority = high_priority ? greatest_priority : least_priority;
        KMM_CUDA_CHECK(cuStreamCreateWithPriority(&stream, flags, priority));
    }

    auto inserted = m_impl->streams.emplace(
        cuda_stream_id(stream),
        DeviceStream::create(context, stream, /*destroy_if_done=*/true)
    );
    return inserted.first->second;
}

void DeviceStreamRegistry::make_progress() {
    std::lock_guard guard {m_impl->mutex};
    for (auto& state : m_impl->streams) {
        state.second.make_progress();
    }
}

void DeviceStreamRegistry::synchronize() {
    std::lock_guard guard {m_impl->mutex};
    for (auto& state : m_impl->streams) {
        state.second.synchronize();
    }
}

}  // namespace kmm
