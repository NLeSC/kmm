#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "kmm/core/macros.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

/// Static hardware properties of a single CUDA device, plus its (primary) CUDA context -- in kmm,
/// a "device" and its primary CUDA context are the same thing.
class DeviceInfo {
  public:
    static constexpr size_t NUM_ATTRIBUTES = G_DEVICE_ATTRIBUTE_MAX;

    DeviceInfo(DeviceId id, g_context_t context);

    /**
     * Returns the name of the device as provided by `gpuDeviceGetName`.
     */
    std::string name() const {
        return m_name;
    }

    /**
     * Returns which memory this device has affinity to.
     */
    MemoryId memory_id() const {
        return MemoryId::device(m_id);
    }

    /**
     * Return this device as a `DeviceId`.
     */
    DeviceId device_id() const {
        return m_id;
    }

    /**
     * Return this device as a `GPUdevice`.
     */
    g_device_t device_ordinal() const {
        return m_device;
    }

    /**
     * Return the (primary) CUDA context of this device.
     */
    g_context_t context() const {
        return m_context;
    }

    /**
     * Return the (primary) CUDA context of this device.
     */
    GPUContextId context_id() const {
        return m_context_id;
    }

    /**
     * Returns the total memory size of this device.
     */
    size_t total_memory_size() const {
        return m_memory_capacity;
    }

    /**
     * Returns the maximum block size supported by this device.
     */
    dim3 max_block_dim() const;

    /**
     * Returns the maximum grid size supported by this device.
     */
    dim3 max_grid_dim() const;

    /**
     * Returns the compute capability of this device as integer (Major, Minor)
     */
    std::pair<int, int> compute_capability() const;

    /**
     * Returns the maximum number of threads per block supported by this device.
     */
    int max_threads_per_block() const;

    /**
     * Returns the value of the provided attribute.
     */
    int attribute(g_device_attribute_t attrib) const;

  private:
    DeviceId m_id;
    g_device_t m_device;
    g_context_t m_context;
    GPUContextId m_context_id;
    std::string m_name;
    size_t m_memory_capacity;
    size_t m_total_memory;
    int m_compute_capability_major;
    int m_compute_capability_minor;
};

/// Static hardware topology of the machine: the number of CUDA devices and their properties.
/// Queried once (via the CUDA driver API) at construction and never changes afterwards. At most
/// `MAX_DEVICES` devices are reported, even if more are physically present, since `DeviceId`
/// itself cannot represent more.
///
/// Retains each device's primary CUDA context for the lifetime of this object (released again on
/// destruction), so this also doubles as the map from a CUDA context back to its `DeviceId` --
/// see `device_id`.
class SystemInfo {
    KMM_NOT_COPYABLE_OR_MOVABLE(SystemInfo)

  public:
    SystemInfo();
    SystemInfo(std::vector<DeviceInfo> devices);
    ~SystemInfo();

    /**
     * Returns the number of GPUs in the system.
     */
    size_t num_devices() const;

    /**
     * Return information on the device with the given identifier.
     */
    const DeviceInfo& device(DeviceId id) const;

    /**
     * Find the device that has the given device ordinal.
     */
    const DeviceInfo& device_by_ordinal(g_device_t ordinal) const;

    /**
     * Returns the highest affinity memory for the given device.
     */
    MemoryId affinity_memory(DeviceId device_id) const;

    /**
     * Returns the device belonging to the given context.
     */
    const DeviceInfo& device_from_context(g_context_t context) const;

    /**
     * Returns the device belonging to the given stream.
     */
    const DeviceInfo& device_from_stream(g_stream_t stream) const;

  private:
    std::vector<DeviceInfo> m_devices;
};

}  // namespace kmm
