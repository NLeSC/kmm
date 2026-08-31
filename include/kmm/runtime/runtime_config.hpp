#pragma once

#include <cstddef>
#include <limits>
#include <string>

namespace kmm {

enum struct HostMemoryKind {
    /// Use a simple memory pool that caches the allocation for the host.
    CachingPool,

    /// No pool. Allocates directly using `cudaMallocHost`.
    NoPool,
};

enum struct DeviceMemoryKind {
    /// Use the default memory pool for the device, as returned by `cudaDeviceGetDefaultMemPool`.
    DefaultPool,

    /// Use a newly created private pool for the device, created using `cudaMemPoolCreate`.
    PrivatePool,

    /// Use a simple memory pool that caches the allocation for the device.
    CachingPool,

    /// No pool. Allocates directly using `cudaMallocAsync` instead of `cudaMallocFromPoolAsync`.
    NoPool,
};

enum struct BufferKind {
    /// One separate allocation per memory, with explicit copies between them when the buffer is
    /// accessed on a memory that does not hold a valid copy. This is the default.
    Discrete,

    /// A single CUDA/HIP managed-memory allocation (`cudaMallocManaged`). The driver migrates
    /// pages between host and devices on demand; KMM never issues explicit copies.
    Managed,

    /// A single pinned host allocation. Devices access it directly (zero-copy) through a mapped
    /// pointer; KMM never stages a device-side copy.
    HostPinned,
};

struct RuntimeConfig {
    /// The type of memory pool to use for the host.
    HostMemoryKind host_memory_kind = HostMemoryKind::NoPool;

    /// Maximum amount of memory that can be allocated on the host, in bytes.
    size_t host_memory_limit = std::numeric_limits<size_t>::max();

    /// If nonzero, use an arena allocator on the host. This will allocate large blocks of the
    /// specified size, which are further split into smaller allocations by the KMM runtime system.
    /// This reduces the number of memory allocation requests to the OS.
    size_t host_memory_block_size = size_t(1024) * 1024 * 100;

    /// The type of memory pool to use on the GPU.
    DeviceMemoryKind device_memory_kind = DeviceMemoryKind::DefaultPool;

    /// Maximum amount of memory that can be allocated on each GPU, in bytes. Note that this
    /// specified limit is allowed to exceed the physical memory size of the GPU, in which case
    /// the physical memory is used as the limit instead.
    size_t device_memory_limit = std::numeric_limits<size_t>::max();

    /// Reserve a specified amount of GPU memory for possibly other applications, in bytes. For example, if this is set
    /// to 1GB and the memory capacity of the GPU is 12 GB, then at most 11 GB will be used by the KMM runtime system
    /// and the remaining space can be used by other applications.
    size_t device_memory_keep_free = 0;

    /// If nonzero, use an arena allocator on each device. This will allocate large blocks of the
    /// specified size, from which smaller allocations are subsequently sub-allocated.
    size_t device_memory_block_size = size_t(1024) * 1024 * 500;

    /// The number of concurrent streams on each device for data transfers.
    size_t device_concurrent_streams = 4;

    /// Enable this run the system in debug mode. This will be significantly slower, but can be
    /// used to track down synchronization bugs.
    bool debug_mode = false;

    /// The buffer kind used by `Runtime::create_buffer` when it is called without an explicit kind.
    BufferKind default_buffer_kind = BufferKind::Discrete;
};

RuntimeConfig default_config_from_environment();

void set_global_log_level(const std::string& name);

}  // namespace kmm