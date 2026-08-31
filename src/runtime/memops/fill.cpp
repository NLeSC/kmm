#include <cstddef>
#include <cstring>

#include "simplify_dims.hpp"

#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/fill.hpp"

namespace kmm {

FillDescription FillDescription::simplify() const {
    FillDescription result = *this;

    for (size_t i = 0; i < result.num_dims; i++) {
        FillDim& dim = result.dims[i];

        // handle negative strides
        if (dim.stride < 0) {
            if (dim.extent > 0) {
                result.offset += (dim.extent - 1) * dim.stride;
            }

            dim.stride = -dim.stride;
        }

        // handle negative extents
        if (dim.extent < 0) {
            dim.extent = 0;
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
