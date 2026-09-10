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

/// The raw bytes of some POD type. For example. use `FillValue::from(int(5))` to construct a `FillValue`.
struct FillValue {
    static constexpr size_t MAX_FILL_LENGTH = 32;

    FillValue() = default;

    template<typename T>
    static FillValue from(T value) {
        static_assert(sizeof(T) <= MAX_FILL_LENGTH, "T exceeds capacity of FillValue");
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

/// Describes a single axis of a strided fill.
struct FillDim {
    memops_extent_type extent = 1;  // number of elements
    memops_stride_type stride = 0;  // step size between elements in bytes.
};

/// Describes a strided multi-dimensional fill of a destination buffer with a repeating element value.
///
/// The destination is treated as an opaque byte buffer: `fill`/`fill_gpu` do not interpret the
/// bytes being written, so `element_size` only determines the width of the value written at each
/// position (see `fill`/`fill_gpu`).
struct FillDescription {
    FillValue value;

    /// Offset add to `dst_addr`, in bytes.
    memops_stride_type offset = 0;

    /// The number of axes described by `dims`. Must be at most `MEMOPS_MAX_DIMS`.
    size_t num_dims = 0;

    /// The extent and per-axis stride.
    FillDim dims[MEMOPS_MAX_DIMS] = {};

    FillDescription() = default;

    explicit FillDescription(FillValue value) : value(value) {}

    /// Appends an axis to this description. `stride` must be given in bytes.
    void add_dimension(memops_extent_type extent, memops_stride_type stride);

    /// Returns an equivalent description with `dims` sorted from the largest stride to the
    /// smallest, and adjacent axes merged whenever they are contiguous (i.e. the outer axis's
    /// stride equals the inner axis's stride times its extent). This reduces the number of
    /// dimensions and may make certain operations easier.
    FillDescription simplify() const;

    /// Returns the total number of elements written (the product of the extent of each axis).
    memops_extent_type num_elements() const {
        memops_extent_type result = 1;

        for (size_t i = 0; i < num_dims; i++) {
            result *= dims[i].extent > 0 ? dims[i].extent : memops_extent_type {0};
        }

        return result;
    }

    /// Returns `true` if this fill writes nothing because some axis has extent zero.
    bool is_empty() const {
        return num_elements() == 0;
    }
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
