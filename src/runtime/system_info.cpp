#include <array>
#include <stdexcept>
#include <utility>

#include "kmm/runtime/system_info.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

DeviceInfo::DeviceInfo(DeviceId id, g_context_t context) :
    m_id(id),
    m_context(context),
    m_context_id(context) {
    GPUContextGuard guard {context};

    KMM_GPU_CHECK(g_ctx_get_device(&m_device));

    std::array<char, 256> name_buf {};
    KMM_GPU_CHECK(g_device_get_name(name_buf.data(), static_cast<int>(name_buf.size()), m_device));
    m_name = name_buf.data();

    KMM_GPU_CHECK(g_device_total_mem(&m_total_memory, m_device));
    m_memory_capacity = m_total_memory;

    KMM_GPU_CHECK(g_device_get_attribute(
        &m_compute_capability_major,
        G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        m_device
    ));
    KMM_GPU_CHECK(g_device_get_attribute(
        &m_compute_capability_minor,
        G_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        m_device
    ));
}

int DeviceInfo::attribute(g_device_attribute_t attrib) const {
    KMM_ASSERT(static_cast<size_t>(attrib) < NUM_ATTRIBUTES);

    int value;
    KMM_GPU_CHECK(g_device_get_attribute(&value, attrib, m_device));
    return value;
}

int DeviceInfo::max_threads_per_block() const {
    return attribute(G_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

dim3 DeviceInfo::max_block_dim() const {
#if defined(KMM_USE_CUDA) || defined(KMM_USE_HIP)
    return dim3(
        static_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
        static_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
        static_cast<unsigned int>(attribute(G_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
    );
#else
    throw std::runtime_error("unsupported operation");
#endif
}

dim3 DeviceInfo::max_grid_dim() const {
#if defined(KMM_USE_CUDA)
    return dim3(
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)),
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)),
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))
    );
#elif defined(KMM_USE_HIP)
    return dim3(
        static_cast<unsigned int>(attribute(HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
        static_cast<unsigned int>(attribute(HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
        static_cast<unsigned int>(attribute(HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
    );
#else
    throw std::runtime_error("unsupported operation");
#endif
}

std::pair<int, int> DeviceInfo::compute_capability() const {
    return {m_compute_capability_major, m_compute_capability_minor};
}

static std::vector<DeviceInfo> query_all_devices() {
    KMM_GPU_CHECK(g_init(0));

    int count = 0;
    KMM_GPU_CHECK(g_device_get_count(&count));

    size_t n = static_cast<size_t>(count) < MAX_DEVICES ? static_cast<size_t>(count) : MAX_DEVICES;

    std::vector<DeviceInfo> devices;
    devices.reserve(n);

    for (size_t i = 0; i < n; i++) {
        g_device_t ordinal;
        KMM_GPU_CHECK(g_device_get(&ordinal, static_cast<int>(i)));

        g_context_t context;
        KMM_GPU_CHECK(g_device_primary_ctx_retain(&context, ordinal));

        devices.emplace_back(DeviceId(i), context);
    }

    return devices;
}

SystemInfo::SystemInfo() : SystemInfo(query_all_devices()) {}

SystemInfo::SystemInfo(std::vector<DeviceInfo> devices) : m_devices(std::move(devices)) {}

SystemInfo::~SystemInfo() {
    // Never throw out of a destructor: ignore release failures rather than routing them through
    // KMM_GPU_CHECK.
    for (const auto& info : m_devices) {
        g_device_primary_ctx_release(info.device_ordinal());
    }
}

size_t SystemInfo::num_devices() const {
    return m_devices.size();
}

const DeviceInfo& SystemInfo::device(DeviceId id) const {
    KMM_ASSERT(id.get() < m_devices.size());
    return m_devices[id.get()];
}

const DeviceInfo& SystemInfo::device_by_ordinal(g_device_t ordinal) const {
    for (const auto& info : m_devices) {
        if (info.device_ordinal() == ordinal) {
            return info;
        }
    }

    throw std::runtime_error("no device found with the given device ordinal");
}

MemoryId SystemInfo::affinity_memory(DeviceId device_id) const {
    return device(device_id).memory_id();
}

const DeviceInfo& SystemInfo::device_from_context(g_context_t context) const {
    for (const auto& info : m_devices) {
        if (info.context() == context) {
            return info;
        }
    }

    throw std::runtime_error("no device found with the given CUDA context");
}

const DeviceInfo& SystemInfo::device_from_stream(g_stream_t stream) const {
    return device_from_context(context_from_stream(stream));
}

}  // namespace kmm
