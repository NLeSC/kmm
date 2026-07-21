
#include "spdlog/spdlog.h"

#include "kmm/memops/gpu_fill.hpp"
#include "kmm/utils/gpu_utils.hpp"
#include "kmm/utils/integer_fun.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

template<typename T, uint32_t block_size>
__global__ void fill_kernel(size_t nelements, T* dest_buffer, T fill_value) {
    size_t i = blockIdx.x * size_t(block_size) + threadIdx.x;

    while (i < nelements) {
        dest_buffer[i] = fill_value;
        i += size_t(block_size) * gridDim.x;
    }
}

template<typename T>
void submit_fill_kernel(
    g_stream_t stream,
    g_device_ptr_t dest_buffer,
    size_t nelements,
    const void* fill_pattern
) {
    static constexpr uint32_t max_grid_size = 512;
    static constexpr uint32_t block_size = 256;

    T fill_value;
    ::memcpy(&fill_value, fill_pattern, sizeof(T));

    uint32_t grid_size = nelements < max_grid_size * size_t(block_size)
        ? static_cast<uint32_t>(div_ceil(nelements, size_t(block_size)))
        : max_grid_size;

    fill_kernel<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(nelements, (T*)dest_buffer, fill_value);
}

template<size_t N>
bool is_fill_pattern_repetitive(const void* fill_pattern, size_t fill_pattern_size) {
    if (fill_pattern_size % N != 0) {
        return false;
    }

    for (size_t i = 1; i < fill_pattern_size / N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (static_cast<const uint8_t*>(fill_pattern)[i * N + j]
                != static_cast<const uint8_t*>(fill_pattern)[j]) {
                return false;
            }
        }
    }

    return true;
}

void execute_gpu_fill_async(g_stream_t stream, g_device_ptr_t dest_buffer, const FillDef& fill) {
    size_t element_size = fill.fill_value.size();
    size_t nbytes = fill.num_elements * element_size;
    const void* fill_pattern = fill.fill_value.data();

    if (nbytes == 0 || element_size == 0) {
        return;
    }

    if (is_fill_pattern_repetitive<1>(fill_pattern, element_size)) {
        uint8_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint8_t));
        KMM_GPU_CHECK(g_memset_d8_async(  //
            g_device_ptr_t(dest_buffer),
            pattern,
            nbytes,
            stream
        ));

    } else if (is_fill_pattern_repetitive<2>(fill_pattern, element_size)) {
        uint16_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint16_t));
        KMM_GPU_CHECK(g_memset_d16_async(  //
            g_device_ptr_t(dest_buffer),
            pattern,
            nbytes / sizeof(uint16_t),
            stream
        ));

    } else if (is_fill_pattern_repetitive<4>(fill_pattern, element_size)) {
        uint32_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint32_t));
        KMM_GPU_CHECK(g_memset_d32_async(  //
            g_device_ptr_t(dest_buffer),
            pattern,
            nbytes / sizeof(uint32_t),
            stream
        ));

    } else if (is_fill_pattern_repetitive<8>(fill_pattern, element_size)) {
        KMM_ASSERT((unsigned long long)(dest_buffer) % 8 == 0);  // must be aligned?
        submit_fill_kernel<uint64_t>(stream, dest_buffer, nbytes / sizeof(uint64_t), fill_pattern);
    } else {
        throw GPUException(fmt::format(
            "could not fill buffer, value is {} bits, but only 8, 16, 32 or 64 bit is supported",
            element_size * 8
        ));
    }
}

}  // namespace kmm
