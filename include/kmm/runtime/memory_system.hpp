#pragma once

#include "kmm/runtime/allocators/base.hpp"
#include "kmm/runtime/stream_manager.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class MemorySystem {
  public:
    virtual ~MemorySystem() = default;

    virtual void make_progress() {}
    virtual void trim_host(size_t bytes_remaining = 0) {}
    virtual void trim_device(size_t bytes_remaining = 0) {}

    virtual AllocationResult allocate_host(
        size_t nbytes,
        DeviceId device_affinity,
        void** ptr_out,
        DeviceEventSet& deps_out
    ) = 0;

    virtual void deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps) = 0;

    virtual AllocationResult allocate_device(
        DeviceId device_id,
        size_t nbytes,
        g_device_ptr_t* ptr_out,
        DeviceEventSet& deps_out
    ) = 0;

    virtual void deallocate_device(
        DeviceId device_id,
        g_device_ptr_t ptr,
        size_t nbytes,
        DeviceEventSet deps
    ) = 0;

    virtual DeviceEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        g_device_ptr_t dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) = 0;

    virtual DeviceEvent copy_device_to_host(
        DeviceId device_id,
        g_device_ptr_t src_addr,
        void* dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) = 0;

    virtual DeviceEvent copy_device_to_device(
        DeviceId src_device,
        DeviceId dst_device,
        g_device_ptr_t src_addr,
        g_device_ptr_t dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) = 0;

    virtual bool is_copy_supported(MemoryId src, MemoryId dst) {
        return true;
    }
};

class MemorySystemImpl: public MemorySystem {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemorySystemImpl)

  public:
    MemorySystemImpl(
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::vector<GPUContextHandle> device_contexts,
        std::unique_ptr<AsyncAllocator> host_mem,
        std::vector<std::unique_ptr<AsyncAllocator>> device_mem
    );

    ~MemorySystemImpl();

    void make_progress();
    void trim_host(size_t bytes_remaining = 0);
    void trim_device(size_t bytes_remaining = 0);

    AllocationResult allocate_host(
        size_t nbytes,
        DeviceId device_affinity,
        void** ptr_out,
        DeviceEventSet& deps_out
    ) final;
    void deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps) final;

    AllocationResult allocate_device(
        DeviceId device_id,
        size_t nbytes,
        g_device_ptr_t* ptr_out,
        DeviceEventSet& deps_out
    ) final;

    void deallocate_device(
        DeviceId device_id,
        g_device_ptr_t ptr,
        size_t nbytes,
        DeviceEventSet deps
    ) final;

    DeviceEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        g_device_ptr_t dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) final;

    DeviceEvent copy_device_to_host(
        DeviceId device_id,
        g_device_ptr_t src_addr,
        void* dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) final;

    DeviceEvent copy_device_to_device(
        DeviceId src_device_id,
        DeviceId dst_device_id,
        g_device_ptr_t src_addr,
        g_device_ptr_t dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) final;

  private:
    struct Device;
    std::shared_ptr<DeviceStreamManager> m_streams;
    std::unique_ptr<AsyncAllocator> m_host;
    std::unique_ptr<Device> m_devices[MAX_DEVICES];
};
}  // namespace kmm
