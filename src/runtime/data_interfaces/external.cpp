#include "fmt/format.h"

#include "kmm/core/panic.hpp"
#include "kmm/runtime/data_interfaces/external.hpp"

namespace kmm {

ExternalDataInterface::ExternalDataInterface(void* ptr, size_t size_in_bytes, MemoryId memory_id) :
    m_ptr(ptr),
    m_size_in_bytes(size_in_bytes),
    m_memory_id(memory_id) {
    KMM_ASSERT(ptr != nullptr);
}

void ExternalDataInterface::check_memory_id(MemoryId memory_id) const {
    if (memory_id != m_memory_id) {
        throw std::runtime_error(
            fmt::format(
                "cannot access or copy buffer on memory {}: buffer is pinned to memory {}",
                memory_id,
                m_memory_id
            )
        );
    }
}

size_t ExternalDataInterface::size_in_bytes() const noexcept {
    return m_size_in_bytes;
}

AllocResult ExternalDataInterface::allocate(
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    DeviceEventSet& deps_out
) {
    check_memory_id(memory_id);
    return AllocResult::Success;
}

void ExternalDataInterface::deallocate(
    MemoryId memory_id,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps
) {
    check_memory_id(memory_id);
}

void* ExternalDataInterface::address(MemoryId memory_id) const noexcept {
    // `allocate` already rejected any other memory, so this is just a sanity check.
    KMM_ASSERT(memory_id == m_memory_id);
    return m_ptr;
}

void ExternalDataInterface::copy(
    MemoryId src,
    MemoryId dst,
    const DeviceStreamId& stream_hint,
    const DeviceEventSet& deps,
    DeviceEventSet& deps_out
) {
    check_memory_id(src);
    check_memory_id(dst);
    deps_out.insert(deps);
}

bool ExternalDataInterface::is_copy_supported(MemoryId src, MemoryId dst) const noexcept {
    return src == m_memory_id && dst == m_memory_id;
}

}  // namespace kmm
