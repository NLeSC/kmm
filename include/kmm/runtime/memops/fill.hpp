#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "kmm/core/macros.hpp"
#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/types.hpp"

namespace kmm {

/// \addtogroup memops
/// @{

/// The raw bytes of a value (e.g. `T{0}`, `T{1}`) to broadcast across a buffer the first time it
/// is materialized in some memory. Empty means "leave the buffer uninitialized" (the default).
struct FillValue {
    static constexpr size_t MAX_FILL_LENGTH = 32;

    FillValue() = default;

    template<typename T>
    static FillValue from(T value) {
        static_assert(sizeof(T) <= MAX_FILL_LENGTH, "T is too large to store in a FillValue");
        static_assert(
            std::is_trivially_copyable_v<T>,
            "T must be trivially copyable to be stored in a FillValue"
        );

        FillValue result;
        result.length = sizeof(T);
        std::memcpy(result.buffer, &value, sizeof(T));
        return result;
    }

    size_t length = 0;
    std::byte buffer[MAX_FILL_LENGTH] {};
};

/// Describes a single axis of a strided fill: `extent` elements are written, and each successive
/// element is offset by `stride` bytes in the destination buffer.
struct FillDim {
    memops_extent_type extent = 1;
    memops_stride_type stride = 0;
};

/// Describes a (possibly strided, possibly multi-dimensional) fill of a destination buffer with
/// a repeating element value.
///
/// The destination is treated as an opaque byte buffer: `fill`/`fill_gpu` do not interpret the
/// bytes being written, so `element_size` only determines the width of the value written at each
/// position (see `fill`/`fill_gpu`).
struct FillDescription {
    FillValue value;

    /// A byte offset added to `dst_addr` before applying `dims`.
    memops_stride_type offset = 0;

    /// The number of axes described by `dims`. Must be at most `MEMOPS_MAX_DIMS`.
    size_t num_dims = 0;

    /// The extent and per-axis stride, ordered from the outermost to the innermost axis.
    FillDim dims[MEMOPS_MAX_DIMS] = {};

    FillDescription() = default;

    KMM_HOST_DEVICE
    explicit FillDescription(FillValue value) : value(value) {}

    /// Appends an axis to this description. `stride` must be given in bytes.
    ///
    /// An axis of extent one is dropped (it is visited exactly once, so its stride never
    /// contributes to addressing). Otherwise, this axis is checked against every existing axis
    /// for contiguity and folded into the first one it is contiguous with instead of consuming a
    /// new slot. This keeps `num_dims` from growing unnecessarily, which matters since `dims`
    /// only has room for `MEMOPS_MAX_DIMS` axes.
    KMM_HOST_DEVICE
    void add_dimension(memops_extent_type extent, memops_stride_type stride) {
        if (extent == 1) {
            return;
        }

        for (size_t i = 0; i < num_dims; i++) {
            FillDim& dim = dims[i];

            // `dim` is the outer neighbor of the new axis: it keeps its extent, but adopts the
            // new axis's (smaller) stride as its own.
            if (dim.stride == stride * extent) {
                dim.extent *= extent;
                dim.stride = stride;
                return;
            }

            // `dim` is the inner neighbor of the new axis: the new axis's stride already
            // matches `dim`'s span, so only `dim`'s extent needs to grow.
            if (stride == dim.stride * dim.extent) {
                dim.extent *= extent;
                return;
            }
        }

        KMM_ASSERT(num_dims < MEMOPS_MAX_DIMS);
        dims[num_dims] = FillDim {extent, stride};
        num_dims++;
    }

    /// Returns the total number of elements written (the product of the extent of each axis).
    KMM_HOST_DEVICE
    memops_extent_type num_elements() const {
        memops_extent_type result = 1;

        for (size_t i = 0; i < num_dims; i++) {
            result *= dims[i].extent;
        }

        return result;
    }

    /// Returns an equivalent description with `dims` sorted from the largest stride to the
    /// smallest, and adjacent axes merged whenever they are contiguous (i.e. the outer axis's
    /// stride equals the inner axis's stride times its extent). This can reduce `num_dims`,
    /// which matters since backends (e.g. `fill_gpu`) only special-case a small number of axes.
    FillDescription simplify() const;
};

/// @}

namespace memops {

/// \addtogroup memops
/// @{

/// Fills `dst_addr` on the CPU, according to `description`, with copies of the `element_size`
/// bytes pointed to by `fill_value`. Blocks until the fill has completed.
void fill(void* dst_addr, const FillDescription& description);

/// @}

}  // namespace memops

}  // namespace kmm
