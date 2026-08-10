#include <array>
#include <stdexcept>
#include <utility>

#include "kmm/runtime/system_info.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

DeviceInfo::DeviceInfo(DeviceId id, CUcontext context) :
    m_id(id),
    m_context(context),
    m_context_id(context) {
    CUDAContextGuard guard {context};

    KMM_CUDA_CHECK(cuCtxGetDevice(&m_device));

    std::array<char, 256> name_buf {};
    KMM_CUDA_CHECK(cuDeviceGetName(name_buf.data(), static_cast<int>(name_buf.size()), m_device));
    m_name = name_buf.data();

    KMM_CUDA_CHECK(cuDeviceTotalMem(&m_total_memory, m_device));
    m_memory_capacity = m_total_memory;

    KMM_CUDA_CHECK(cuDeviceGetAttribute(
        &m_compute_capability_major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        m_device
    ));
    KMM_CUDA_CHECK(cuDeviceGetAttribute(
        &m_compute_capability_minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        m_device
    ));
}

int DeviceInfo::attribute(CUdevice_attribute attrib) const {
    KMM_ASSERT(static_cast<size_t>(attrib) < NUM_ATTRIBUTES);

    int value;
    KMM_CUDA_CHECK(cuDeviceGetAttribute(&value, attrib, m_device));
    return value;
}

int DeviceInfo::max_threads_per_block() const {
    return attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

dim3 DeviceInfo::max_block_dim() const {
    return dim3(
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
    );
}

dim3 DeviceInfo::max_grid_dim() const {
    return dim3(
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)),
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)),
        static_cast<unsigned int>(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))
    );
}

std::pair<int, int> DeviceInfo::compute_capability() const {
    return {m_compute_capability_major, m_compute_capability_minor};
}

static std::vector<DeviceInfo> query_all_devices() {
    KMM_CUDA_CHECK(cuInit(0));

    int count = 0;
    KMM_CUDA_CHECK(cuDeviceGetCount(&count));

    size_t n = static_cast<size_t>(count) < MAX_DEVICES ? static_cast<size_t>(count) : MAX_DEVICES;

    std::vector<DeviceInfo> devices;
    devices.reserve(n);

    for (size_t i = 0; i < n; i++) {
        CUdevice ordinal;
        KMM_CUDA_CHECK(cuDeviceGet(&ordinal, static_cast<int>(i)));

        CUcontext context;
        KMM_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, ordinal));

        devices.emplace_back(DeviceId(i), context);
    }

    return devices;
}

SystemInfo::SystemInfo() : SystemInfo(query_all_devices()) {}

SystemInfo::SystemInfo(std::vector<DeviceInfo> devices) : m_devices(std::move(devices)) {}

SystemInfo::~SystemInfo() {
    // Never throw out of a destructor: ignore release failures rather than routing them through
    // KMM_CUDA_CHECK.
    for (const auto& info : m_devices) {
        cuDevicePrimaryCtxRelease(info.device_ordinal());
    }
}

size_t SystemInfo::num_devices() const {
    return m_devices.size();
}

const DeviceInfo& SystemInfo::device(DeviceId id) const {
    KMM_ASSERT(id.get() < m_devices.size());
    return m_devices[id.get()];
}

const DeviceInfo& SystemInfo::device_by_ordinal(CUdevice ordinal) const {
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

const DeviceInfo& SystemInfo::device_from_context(CUcontext context) const {
    for (const auto& info : m_devices) {
        if (info.context() == context) {
            return info;
        }
    }

    throw std::runtime_error("no device found with the given CUDA context");
}

const DeviceInfo& SystemInfo::device_from_stream(CUstream stream) const {
    CUcontext context;
    KMM_CUDA_CHECK(cuStreamGetCtx(stream, &context));
    return device_from_context(context);
}

}  // namespace kmm
