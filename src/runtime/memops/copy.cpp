#include <cstddef>
#include <cstring>

#include "simplify_dims.hpp"

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/copy.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

static void copy_dim(
    const std::byte* src,
    std::byte* dst,
    const CopyDim* dims,
    size_t num_dims,
    size_t element_size
) {
    if (num_dims == 0) {
        std::memcpy(dst, src, element_size);
        return;
    }

    for (memops_extent_type i = 0; i < dims->extent; i++) {
        copy_dim(
            src + i * dims->src_stride,
            dst + i * dims->dst_stride,
            dims + 1,
            num_dims - 1,
            element_size
        );
    }
}

CopyDescription CopyDescription::simplify() const {
    CopyDescription result = *this;

    result.num_dims = simplify_dims(
        dims,
        num_dims,
        result.dims,
        [](const CopyDim& a, const CopyDim& b) {
            // Descending stride order, ties broken by `dst_stride`.
            return a.src_stride != b.src_stride ? a.src_stride > b.src_stride
                                                 : a.dst_stride > b.dst_stride;
        },
        [](CopyDim& outer, const CopyDim& inner) {
            if (outer.src_stride == inner.src_stride * inner.extent
                && outer.dst_stride == inner.dst_stride * inner.extent) {
                outer.extent *= inner.extent;
                outer.src_stride = inner.src_stride;
                outer.dst_stride = inner.dst_stride;
                return true;
            }

            return false;
        }
    );

    // The innermost axis (last, since `dims` is sorted in descending stride order) may itself be
    // contiguous with the element: if its stride on both sides equals `element_size`, it can be
    // folded into `element_size` instead of being kept as a separate axis.
    while (result.num_dims > 0
           && is_equal(result.dims[result.num_dims - 1].src_stride, result.element_size)
           && is_equal(result.dims[result.num_dims - 1].dst_stride, result.element_size)) {
        result.element_size *= checked_cast<size_t>(result.dims[result.num_dims - 1].extent);
        result.num_dims--;
    }

    return result;
}

void copy(const void* src_addr, void* dst_addr, const CopyDescription& description) {
    copy_dim(
        static_cast<const std::byte*>(src_addr),
        static_cast<std::byte*>(dst_addr),
        description.dims,
        description.num_dims,
        description.element_size
    );
}

void copy_async(
    CUstream stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
) {
    size_t n = description.num_dims;
    size_t element_size = description.element_size;

    if (n == 0) {
        // No axes: a single contiguous run of `element_size` bytes.
        KMM_CUDA_CHECK(cuMemcpyAsync(
            (CUdeviceptr)dst_addr,
            (CUdeviceptr)src_addr,
            element_size,
            stream
        ));
    } else if (n == 1) {
        const CopyDim& dim = description.dims[0];

        if (dim.src_stride == static_cast<memops_stride_type>(element_size)
            && dim.dst_stride == static_cast<memops_stride_type>(element_size)) {
            // Both sides are contiguous: the whole axis collapses into one flat copy.
            KMM_CUDA_CHECK(cuMemcpyAsync(
                (CUdeviceptr)dst_addr,
                (CUdeviceptr)src_addr,
                element_size * static_cast<size_t>(dim.extent),
                stream
            ));
        } else {
            // One strided axis on top of a contiguous element: a 2D pitched copy, where
            // each "row" is a single element and `dim.extent` rows are copied.
            CUDA_MEMCPY2D copy_params;
            std::memset(&copy_params, 0, sizeof(copy_params));

            copy_params.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
            copy_params.srcDevice = (CUdeviceptr)src_addr;
            copy_params.srcPitch = static_cast<size_t>(dim.src_stride);

            copy_params.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
            copy_params.dstDevice = (CUdeviceptr)dst_addr;
            copy_params.dstPitch = static_cast<size_t>(dim.dst_stride);

            copy_params.WidthInBytes = element_size;
            copy_params.Height = static_cast<size_t>(dim.extent);

            KMM_CUDA_CHECK(cuMemcpy2DAsync(&copy_params, stream));
        }
    } else {
        // Copies with two or more independently-strided batch axes do not map onto the
        // driver API's pitched `cuMemcpy2D`/`cuMemcpy3D` calls in general (those assume a
        // regularly-nested pitch, not arbitrary independent strides per axis). Supporting
        // this fully requires a small copy kernel, which is not implemented yet.
        KMM_TODO();
    }
}

}  // namespace kmm
