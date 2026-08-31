#include <cstddef>
#include <cstring>

#include "simplify_dims.hpp"

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/copy.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

static Range<ptrdiff_t> dim_offset_range(
    ptrdiff_t base_offset,
    const CopyDim* dims,
    size_t num_dims,
    size_t element_size,
    memops_stride_type CopyDim::*stride_member
) {
    ptrdiff_t lo = base_offset;
    ptrdiff_t hi = base_offset;

    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i].extent < 1) {
            return {base_offset, base_offset};
        }

        ptrdiff_t span = checked_mul<ptrdiff_t>(dims[i].extent - 1, dims[i].*stride_member);
        lo = checked_add(lo, span < 0 ? span : 0);
        hi = checked_add(hi, span > 0 ? span : 0);
    }

    return {lo, checked_add<ptrdiff_t>(hi, element_size)};
}

Range<ptrdiff_t> CopyDescription::src_range() const {
    return dim_offset_range(
        static_cast<ptrdiff_t>(src_offset),
        dims,
        num_dims,
        element_size,
        &CopyDim::src_stride
    );
}

Range<ptrdiff_t> CopyDescription::dst_range() const {
    return dim_offset_range(
        static_cast<ptrdiff_t>(dst_offset),
        dims,
        num_dims,
        element_size,
        &CopyDim::dst_stride
    );
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

void copy(const void* src_addr, void* dst_addr, const CopyDescription& description) {
    copy_dim(
        static_cast<const std::byte*>(src_addr) + description.src_offset,
        static_cast<std::byte*>(dst_addr) + description.dst_offset,
        description.dims,
        description.num_dims,
        description.element_size
    );
}

void copy_async(
    GPUStream stream,
    const void* src_addr,
    void* dst_addr,
    const CopyDescription& description
) {
    src_addr = static_cast<const std::byte*>(src_addr) + description.src_offset;
    dst_addr = static_cast<std::byte*>(dst_addr) + description.dst_offset;

    size_t n = description.num_dims;
    size_t element_size = description.element_size;
    if (n == 0) {
        // No axes: a single contiguous run of `element_size` bytes.
        KMM_GPU_CHECK(gpuMemcpyDtoDAsync(
            (GPUDeviceptr)dst_addr,
            (GPUDeviceptr)src_addr,
            element_size,
            stream
        ));

        return;
    }

#if defined(KMM_USE_CUDA)
    if (n == 1) {
        const CopyDim& dim = description.dims[0];

        // One strided axis on top of a contiguous element: a 2D pitched copy, where
        // each "row" is a single element and `dim.extent` rows are copied.
        CUDA_MEMCPY2D copy_params;
        std::memset(&copy_params, 0, sizeof(copy_params));

        copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy_params.srcDevice = (GPUDeviceptr)src_addr;
        copy_params.srcPitch = static_cast<size_t>(dim.src_stride);

        copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy_params.dstDevice = (GPUDeviceptr)dst_addr;
        copy_params.dstPitch = static_cast<size_t>(dim.dst_stride);

        copy_params.WidthInBytes = element_size;
        copy_params.Height = static_cast<size_t>(dim.extent);

        KMM_GPU_CHECK(gpuMemcpy2DAsync(&copy_params, stream));
        return;
    }
#endif

    throw std::runtime_error("copy operation unsupported");
}

}  // namespace kmm
