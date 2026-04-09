#pragma once

#include <array>
#include <string>
#include <vector>

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

class DeviceInfo {
  public:
    static constexpr size_t NUM_ATTRIBUTES = G_DEVICE_ATTRIBUTE_MAX;

    DeviceInfo(DeviceId id, GPUContextHandle context, size_t num_concurrent_streams = 1);

    /**
     * Returns the name of the device as provided by `g_device_get_name`.
     */
    std::string name() const {
        return m_name;
    }

    /**
     * Returns which memory this device has affinity to.
     */
    MemoryId memory_id() const {
        return MemoryId(m_id);
    }

    /**
     * Return this device as a `DeviceId`.
     */
    DeviceId device_id() const {
        return m_id;
    }

    /**
     * Return this device as a `g_device_t`.
     */
    g_device_t device_ordinal() const {
        return m_device_id;
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
     * Returns the compute capability of this device as integer `MAJOR * 10 + MINOR` (For example,
     * `86` means capability 8.6)
     */
    int compute_capability() const;

    /**
     * Returns the maximum number of threads per block supported by this device.
     */
    int max_threads_per_block() const;

    /**
     * Returns the value of the provided attribute.
     */
    int attribute(g_device_attribute_t attrib) const;

    size_t num_compute_streams() const {
        return m_concurrent_stream;
    }

  private:
    DeviceId m_id;
    std::string m_name;
    g_device_t m_device_id;
    size_t m_memory_capacity;
    size_t m_concurrent_stream = 1;
    std::array<int, NUM_ATTRIBUTES> m_attributes;
};

class SystemInfo {
  public:
    SystemInfo() = default;
    SystemInfo(std::vector<DeviceInfo> devices);
    SystemInfo(SystemInfo info, std::vector<ResourceId> subresources);

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
    const DeviceInfo& device_by_ordinal(g_device_t device_ordinal) const;

    /**
     * Return a list of the available processors in the system.
     */
    std::vector<ResourceId> resources() const;

    /**
     * Return a list of the available memories in the system.
     */
    std::vector<MemoryId> memories() const;

    /**
     * Returns the highest affinity memory for the given processor.
     */
    MemoryId affinity_memory(ResourceId proc_id) const;

    /**
     * Returns the highest affinity memory for the given device.
     */
    MemoryId affinity_memory(DeviceId device_id) const;

    /**
     * Returns the processor that has the highest affinity for accessing the given memory.
     */
    static ResourceId affinity_processor(MemoryId memory_id);

    /**
     * Checks if the given processor can access the given memory.
     */
    bool is_memory_accessible(MemoryId memory_id, ResourceId proc_id) const;

    /**
     * Checks if the given device can access the given memory.
     */
    bool is_memory_accessible(MemoryId memory_id, DeviceId device_id) const;

  private:
    std::vector<DeviceInfo> m_devices;
    std::vector<ResourceId> m_resources;
};

}  // namespace kmm