#include <algorithm>

#include "fmt/format.h"

#include "kmm/core/system_info.hpp"

namespace kmm {

DeviceInfo::DeviceInfo(DeviceId id, GPUContextHandle context, size_t num_concurrent_streams) :
    m_id(id),
    m_concurrent_stream(num_concurrent_streams) {
    GPUContextGuard guard {context};

    KMM_GPU_CHECK(g_ctx_get_device(&m_device_id));

    char name[1024];
    KMM_GPU_CHECK(g_device_get_name(name, 1024, m_device_id));
    m_name = std::string(name);

    for (size_t i = 1; i < NUM_ATTRIBUTES; i++) {
        auto attr = g_device_attribute_t(i);
        KMM_GPU_CHECK(g_device_get_attribute(&m_attributes[i], attr, m_device_id));
    }

    size_t ignore_free_memory;
    KMM_GPU_CHECK(g_mem_get_info(&ignore_free_memory, &m_memory_capacity));
}

dim3 DeviceInfo::max_block_dim() const {
    return dim3(
        checked_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
        checked_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
        checked_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
    );
}

dim3 DeviceInfo::max_grid_dim() const {
    return dim3(
        checked_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)),
        checked_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)),
        checked_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))
    );
}

int DeviceInfo::compute_capability() const {
    return (attribute(G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10)
        + attribute(G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
}

int DeviceInfo::max_threads_per_block() const {
    return attribute(G_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

int DeviceInfo::attribute(g_device_attribute_t attrib) const {
    if (attrib < NUM_ATTRIBUTES) {
        return m_attributes[attrib];
    }

    throw std::runtime_error("unsupported attribute requested");
}

SystemInfo::SystemInfo(std::vector<DeviceInfo> devices) : m_devices(devices) {
    for (const auto& dev : devices) {
        m_resources.push_back(ResourceId(dev.device_id()));
    }
}

SystemInfo::SystemInfo(SystemInfo base, std::vector<ResourceId> subresources) :
    m_devices(base.m_devices),
    m_resources(std::move(subresources)) {
    for (const auto& resource : m_resources) {
        bool is_valid = false;

        for (auto base_resource : base.m_resources) {
            is_valid |= base_resource.contains(resource);
        }

        if (!is_valid) {
            throw std::runtime_error(fmt::format("invalid resource: {}", resource));
        }
    }
}

size_t SystemInfo::num_devices() const {
    return m_devices.size();
}

const DeviceInfo& SystemInfo::device(DeviceId id) const {
    return m_devices.at(id.get());
}

const DeviceInfo& SystemInfo::device_by_ordinal(g_device_t ordinal) const {
    for (const auto& device : m_devices) {
        if (device.device_ordinal() == ordinal) {
            return device;
        }
    }

    throw std::runtime_error(fmt::format("cannot find device with ordinal {}", ordinal));
}

std::vector<ResourceId> SystemInfo::resources() const {
    return m_resources;
}

std::vector<MemoryId> SystemInfo::memories() const {
    std::vector<MemoryId> result {MemoryId::host()};
    for (const auto& device : m_devices) {
        result.push_back(device.memory_id());
    }

    return result;
}

MemoryId SystemInfo::affinity_memory(DeviceId device_id) const {
    return device(device_id).memory_id();
}

MemoryId SystemInfo::affinity_memory(ResourceId proc_id) const {
    if (proc_id.is_device()) {
        return affinity_memory(proc_id.as_device());
    } else {
        return MemoryId::host();
    }
}

ResourceId SystemInfo::affinity_processor(MemoryId memory_id) {
    if (memory_id.is_device()) {
        return memory_id.as_device();
    } else {
        return ResourceId::host();
    }
}

bool SystemInfo::is_memory_accessible(MemoryId memory_id, ResourceId proc_id) const {
    if (!memory_id.is_host() && proc_id.is_device()) {
        return affinity_memory(proc_id.as_device()) == memory_id;
    }

    return memory_id.is_host();
}

bool SystemInfo::is_memory_accessible(MemoryId memory_id, DeviceId device_id) const {
    return is_memory_accessible(memory_id, ResourceId(device_id));
}

}  // namespace kmm
