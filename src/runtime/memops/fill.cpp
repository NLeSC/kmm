#include <cstddef>
#include <cstring>

#include "simplify_dims.hpp"

#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/fill.hpp"

namespace kmm {

void FillDescription::add_dimension(memops_extent_type extent, memops_stride_type stride) {
    // add the dimensions
    if (num_dims < MEMOPS_MAX_DIMS) {
        dims[num_dims] = FillDim {extent, stride};
        num_dims++;
        return;
    }

    // could not add the dimensions, try to fuse it with one of the existing dimensions
    for (auto& dim : this->dims) {
        // `dim` is the outer neighbor of the new axis or has extent 1
        if (dim.stride == stride * extent || dim.extent == 1) {
            dim.extent *= extent;
            dim.stride = stride;
            return;
        }

        // `dim` is the inner neighbor of the new axis
        if (stride == dim.stride * dim.extent) {
            dim.extent *= extent;
            return;
        }
    }

    throw std::runtime_error(
        "cannot add dimension to `FillDescription`, exceeds maximum number of dimensions"
    );
}

FillDescription FillDescription::simplify() const {
    FillDescription result = *this;

    for (size_t i = 0; i < result.num_dims; i++) {
        FillDim& dim = result.dims[i];

        // A zero or negative extent makes the whole fill empty. `simplify_dims` handles this.
        if (dim.extent <= 0) {
            continue;
        }

        // Rewrite negative strides to positive, shifting `offset` to the far end of the axis so
        // the same elements are still visited. (`simplify_dims` handles negative extents.)
        if (dim.stride < 0) {
            result.offset += (dim.extent - 1) * dim.stride;
            dim.stride = -dim.stride;
        }

        // A zero stride revisits the same address `extent` times; for `fill` those repeated
        // writes are redundant, so collapse the axis to a single element and let `simplify_dims`
        // drop it.
        if (dim.stride == 0) {
            dim.extent = 1;
        }
    }

    result.num_dims = simplify_dims(
        result.dims,
        result.num_dims,
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

namespace memops {

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

void fill(void* dst_addr, const FillDescription& description) {
    fill_dim(
        static_cast<std::byte*>(dst_addr) + description.offset,
        description.dims,
        description.num_dims,
        description.value.length,
        description.value.buffer
    );
}

}  // namespace memops

}  // namespace kmm
