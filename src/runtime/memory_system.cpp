#include <utility>

#include "kmm/runtime/memory_system.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

struct MemorySystem::DeviceState {
    DeviceState(CUcontext context, DeviceStreamRegistry* streams, size_t memory_limit) :
        context(context),
        native_stream(context),
        copy_stream(streams->lookup(native_stream.get())),
        allocator(std::make_unique<DeviceMemoryAllocator>(context, memory_limit)) {}

    CUcontext context;
    CUDAStream native_stream;
    DeviceStream copy_stream;
    std::unique_ptr<AsyncAllocator> allocator;
};

MemorySystem::MemorySystem(
    const SystemInfo& system_info,
    DeviceStreamRegistry* streams,
    size_t host_memory_limit,
    size_t device_memory_limit
) :
    m_num_devices(system_info.num_devices()) {
    KMM_ASSERT(m_num_devices > 0);
    KMM_ASSERT(m_num_devices <= MAX_DEVICES);

    // Portable pinned memory is visible to every context, so any device's context works here.
    m_host_allocator = std::make_unique<PinnedMemoryAllocator>(
        system_info.device(DeviceId(0)).context(),
        host_memory_limit
    );

    for (size_t i = 0; i < m_num_devices; i++) {
        auto context = system_info.device(DeviceId(i)).context();
        m_devices[i] = std::make_unique<DeviceState>(context, streams, device_memory_limit);
    }

    // Determine, and where possible enable, peer-to-peer access between every device pair.
    for (size_t i = 0; i < m_num_devices; i++) {
        m_peer_access[i][i] = true;

        for (size_t j = i + 1; j < m_num_devices; j++) {
            int i_can_access_j = 0;
            int j_can_access_i = 0;

            KMM_CUDA_CHECK(cuDeviceCanAccessPeer(
                &i_can_access_j,
                system_info.device(DeviceId(i)).device_ordinal(),
                system_info.device(DeviceId(j)).device_ordinal()
            ));

            KMM_CUDA_CHECK(cuDeviceCanAccessPeer(
                &j_can_access_i,
                system_info.device(DeviceId(j)).device_ordinal(),
                system_info.device(DeviceId(i)).device_ordinal()
            ));

            m_peer_access[i][j] = i_can_access_j != 0;
            m_peer_access[j][i] = j_can_access_i != 0;

            if (i_can_access_j) {
                CUDAContextGuard guard {m_devices[i]->context};
                CUresult result = cuCtxEnablePeerAccess(m_devices[j]->context, 0);

                if (result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                    KMM_CUDA_CHECK(result);
                }
            }

            if (j_can_access_i) {
                CUDAContextGuard guard {m_devices[j]->context};
                CUresult result = cuCtxEnablePeerAccess(m_devices[i]->context, 0);

                if (result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                    KMM_CUDA_CHECK(result);
                }
            }
        }
    }
}

MemorySystem::~MemorySystem() = default;

MemorySystem::DeviceState& MemorySystem::device_state(DeviceId id) const {
    KMM_ASSERT(id.get() < m_num_devices);
    return *m_devices[id.get()];
}

DeviceStream MemorySystem::stream(DeviceId device) const {
    return device_state(device).copy_stream;
}

void MemorySystem::make_progress() {
    m_streams->make_progress();
    m_host_allocator->poll();

    for (size_t i = 0; i < m_num_devices; i++) {
        m_devices[i]->allocator->poll();
    }
}

void MemorySystem::trim_host(size_t bytes_remaining) {
    m_host_allocator->trim(bytes_remaining);
}

void MemorySystem::trim_device(size_t bytes_remaining) {
    for (size_t i = 0; i < m_num_devices; i++) {
        m_devices[i]->allocator->trim(bytes_remaining);
    }
}

AllocResult MemorySystem::allocate_host(
    size_t nbytes,
    DeviceId device_affinity,
    void** ptr_out,
    DeviceEventSet& deps_out
) {
    return m_host_allocator->allocate_async(  //
        device_state(device_affinity).copy_stream,
        BufferLayout {nbytes},
        ptr_out,
        deps_out
    );
}

void MemorySystem::deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps) {
    m_host_allocator
        ->deallocate_async(DeviceStream {}, ptr, BufferLayout {nbytes}, std::move(deps));
}

AllocResult MemorySystem::allocate_device(
    DeviceId device_id,
    size_t nbytes,
    CUdeviceptr* ptr_out,
    DeviceEventSet& deps_out
) {
    auto& dev = device_state(device_id);
    void* addr = nullptr;

    AllocResult result =
        dev.allocator->allocate_async(dev.copy_stream, BufferLayout {nbytes}, &addr, deps_out);

    if (result == AllocResult::Success) {
        *ptr_out = CUdeviceptr(addr);
    }

    return result;
}

void MemorySystem::deallocate_device(
    DeviceId device_id,
    CUdeviceptr ptr,
    size_t nbytes,
    DeviceEventSet deps
) {
    KMM_TODO();
}

DeviceEvent MemorySystem::copy_host_to_device(
    DeviceId device_id,
    const void* src_addr,
    CUdeviceptr dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    auto& dev = device_state(device_id);
    CUDAContextGuard guard {dev.context};

    return dev.copy_stream.with_stream(deps, [&](CUstream stream) {
        KMM_CUDA_CHECK(cuMemcpyHtoDAsync(dst_addr, src_addr, nbytes, stream));
    });
}

DeviceEvent MemorySystem::copy_device_to_host(
    DeviceId device_id,
    CUdeviceptr src_addr,
    void* dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    auto& dev = device_state(device_id);
    CUDAContextGuard guard {dev.context};

    return dev.copy_stream.with_stream(deps, [&](CUstream stream) {
        KMM_CUDA_CHECK(cuMemcpyDtoHAsync(dst_addr, src_addr, nbytes, stream));
    });
}

DeviceEvent MemorySystem::copy_device_to_device(
    DeviceId src_device,
    DeviceId dst_device,
    CUdeviceptr src_addr,
    CUdeviceptr dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    auto& src = device_state(src_device);
    auto& dst = device_state(dst_device);
    CUDAContextGuard guard {dst.context};

    return dst.copy_stream.with_stream(deps, [&](CUstream stream) {
        KMM_CUDA_CHECK(
            cuMemcpyPeerAsync(dst_addr, dst.context, src_addr, src.context, nbytes, stream)
        );
    });
}

bool MemorySystem::is_copy_supported(MemoryId src, MemoryId dst) {
    if (src.is_host() || dst.is_host()) {
        return true;
    }

    auto src_id = src.as_device();
    auto dst_id = dst.as_device();

    if (src_id == dst_id) {
        return true;
    }

    return m_peer_access[src_id.get()][dst_id.get()];
}

}  // namespace kmm
