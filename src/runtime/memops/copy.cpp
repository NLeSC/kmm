#include <cstddef>
#include <cstring>

#include "simplify_dims.hpp"

#include "kmm/core/checked_compare.hpp"
#include "kmm/core/integer_fun.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/copy.hpp"

namespace kmm {

static Range<ptrdiff_t> dim_offset_range(
    ptrdiff_t base_offset,
    const CopyDim* dims,
    size_t num_dims,
    size_t element_size,
    memops_stride_type CopyDim::* stride_member
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

    for (size_t i = 0; i < result.num_dims; i++) {
        CopyDim& dim = result.dims[i];

        // zero extent means no copy at all.
        if (dim.extent <= 0) {
            return CopyDescription {0};
        }

        // Rewrite a negative primary stride to positive
        if (dim.dst_stride < 0) {
            result.src_offset += (dim.extent - 1) * dim.src_stride;
            result.dst_offset += (dim.extent - 1) * dim.dst_stride;

            dim.src_stride = -dim.src_stride;
            dim.dst_stride = -dim.dst_stride;
        }
    }

    result.num_dims = simplify_dims(
        result.dims,
        result.num_dims,
        result.dims,
        [](const CopyDim& a, const CopyDim& b) {
            // Descending stride order, keyed on `dst_stride` (ties broken by `src_stride`): it
            // matters more that writes to the innermost axis coalesce than that reads do.
            return a.dst_stride != b.dst_stride
                ? unsigned_abs(a.dst_stride) > unsigned_abs(b.dst_stride)
                : unsigned_abs(a.src_stride) > unsigned_abs(b.src_stride);
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

namespace memops {

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

}  // namespace memops

}  // namespace kmm
