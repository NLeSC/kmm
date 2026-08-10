#include <cstddef>
#include <cstring>

#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/fill.hpp"
#include "kmm/utils/gpu_utils.hpp"
#include "simplify_dims.hpp"

namespace kmm {

static void fill_dim(
    std::byte* dst,
    const FillDim* dims,
    size_t num_dims,
    size_t element_size,
    const void* fill_value
) {
    if (num_dims == 0) {
        std::memcpy(dst, fill_value, element_size);
        return;
    }

    for (memops_extent_type i = 0; i < dims->extent; i++) {
        fill_dim(dst + i * dims->stride, dims + 1, num_dims - 1, element_size, fill_value);
    }
}

FillDescription FillDescription::simplify() const {
    FillDescription result = *this;

    result.num_dims = simplify_dims(
        dims,
        num_dims,
        result.dims,
        [](const FillDim& a, const FillDim& b) { return a.stride > b.stride; },
        [](FillDim& outer, const FillDim& inner) {
            if (outer.stride == inner.stride * inner.extent) {
                outer.extent *= inner.extent;
                outer.stride = inner.stride;
                return true;
            }

            return false;
        }
    );

    return result;
}

void fill(void* dst_addr, const FillDescription& description, const void* fill_value) {
    fill_dim(
        static_cast<std::byte*>(dst_addr),
        description.dims,
        description.num_dims,
        description.value.length,
        fill_value
    );
}

void fill_async(
    CUstream stream,
    void* dst_addr,
    const FillDescription& description,
    const void* fill_value
) {
    size_t n = description.num_dims;
    size_t element_size = description.value.length;

    // The driver API only offers native (strided) memset support for 1/2/4-byte elements,
    // and only up to one strided axis (`cuMemsetD2D*Async`). Anything wider, or with more
    // than one strided batch axis, requires a small fill kernel, which is not implemented
    // yet.
    if (element_size == 1) {
        unsigned char value;
        std::memcpy(&value, fill_value, sizeof(value));

        if (n == 0) {
            KMM_CUDA_CHECK(cuMemsetD8Async((CUdeviceptr)dst_addr, value, 1, stream));
        } else if (n == 1) {
            const FillDim& dim = description.dims[0];

            if (dim.stride == static_cast<memops_stride_type>(element_size)) {
                KMM_CUDA_CHECK(cuMemsetD8Async(
                    (CUdeviceptr)dst_addr,
                    value,
                    static_cast<size_t>(dim.extent),
                    stream
                ));
            } else {
                KMM_CUDA_CHECK(cuMemsetD2D8Async(
                    (CUdeviceptr)dst_addr,
                    static_cast<size_t>(dim.stride),
                    value,
                    1,
                    static_cast<size_t>(dim.extent),
                    stream
                ));
            }
        } else {
            KMM_TODO();
        }
    } else if (element_size == 2) {
        unsigned short value;
        std::memcpy(&value, fill_value, sizeof(value));

        if (n == 0) {
            KMM_CUDA_CHECK(cuMemsetD16Async((CUdeviceptr)dst_addr, value, 1, stream));
        } else if (n == 1) {
            const FillDim& dim = description.dims[0];

            if (dim.stride == static_cast<memops_stride_type>(element_size)) {
                KMM_CUDA_CHECK(cuMemsetD16Async(
                    (CUdeviceptr)dst_addr,
                    value,
                    static_cast<size_t>(dim.extent),
                    stream
                ));
            } else {
                KMM_CUDA_CHECK(cuMemsetD2D16Async(
                    (CUdeviceptr)dst_addr,
                    static_cast<size_t>(dim.stride),
                    value,
                    1,
                    static_cast<size_t>(dim.extent),
                    stream
                ));
            }
        } else {
            KMM_TODO();
        }
    } else if (element_size == 4) {
        unsigned int value;
        std::memcpy(&value, fill_value, sizeof(value));

        if (n == 0) {
            KMM_CUDA_CHECK(cuMemsetD32Async((CUdeviceptr)dst_addr, value, 1, stream));
        } else if (n == 1) {
            const FillDim& dim = description.dims[0];

            if (dim.stride == static_cast<memops_stride_type>(element_size)) {
                KMM_CUDA_CHECK(cuMemsetD32Async(
                    (CUdeviceptr)dst_addr,
                    value,
                    static_cast<size_t>(dim.extent),
                    stream
                ));
            } else {
                KMM_CUDA_CHECK(cuMemsetD2D32Async(
                    (CUdeviceptr)dst_addr,
                    static_cast<size_t>(dim.stride),
                    value,
                    1,
                    static_cast<size_t>(dim.extent),
                    stream
                ));
            }
        } else {
            KMM_TODO();
        }
    } else {
        KMM_TODO();
    }
}

}  // namespace kmm
