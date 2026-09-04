#pragma once

#include <cstdint>

#include "kmm/core/key_value.hpp"
#include "kmm/core/macros.hpp"

namespace kmm {

/// \addtogroup memops
/// @{

/// The maximum number of dimensions supported by the strided descriptors in `kmm/memops`
/// (`CopyDescription`, `FillDescription`, `ReductionDescription`). Kept small and fixed so these
/// descriptors stay plain, fixed-size, trivially-copyable structs that can be passed by value to
/// GPU kernels.
inline constexpr size_t MEMOPS_MAX_DIMS = 4;

/// The extent (number of elements) along a single axis of a `kmm/memops` descriptor.
using memops_extent_type = signed long long int;

/// A byte offset between consecutive elements along a single axis of a `kmm/memops` descriptor.
using memops_stride_type = signed long long int;

/// A runtime tag for the scalar element type operated on by `fill`/`reduce`. Since the functions
/// in `kmm/memops` operate on type-erased `void*` buffers, they need this at runtime to know how
/// to interpret the bytes at each element (e.g. how to add two elements together for a `Sum`
/// reduction, or how wide the fill pattern is).
enum class DataType {
    Unknown = 0,
    Int32,
    Int64,
    Uint32,
    Uint64,
    Float32,
    Float64,
    /// `KeyValue<int64_t>` / `KeyValue<double>`: a value paired with its `int64_t` key. Only
    /// meaningful for `Min`/`Max` reductions (i.e. argmin/argmax).
    KeyValueInt64,
    KeyValueFloat64,
};

/// Returns the size (in bytes) of a single element of the given data type.
size_t data_type_size(DataType dtype);

/// Returns a human-readable name for the given data type (e.g. `"Float32"`).
const char* data_type_name(DataType dtype);

/// The operator applied by `reduce`/`reduce_gpu` to combine elements. `BitwiseAnd`/`BitwiseOr` are
/// only valid for the integer data types.
enum class ReductionOp {
    Sum,
    Product,
    Min,
    Max,
    BitwiseAnd,
    BitwiseOr,
};

/// Returns a human-readable name for the given reduction operator (e.g. `"Sum"`).
const char* reduction_op_name(ReductionOp op);

template<typename T>
struct data_type_of {};

#define KMM_IMPL_DATA_TYPE_OF(TYPE, DTYPE)    \
    template<>                                \
    struct data_type_of<TYPE> {               \
        constexpr operator DataType() const { \
            return DataType::DTYPE;           \
        }                                     \
    };

KMM_IMPL_DATA_TYPE_OF(int32_t, Int32)
KMM_IMPL_DATA_TYPE_OF(int64_t, Int64)
KMM_IMPL_DATA_TYPE_OF(uint32_t, Uint32)
KMM_IMPL_DATA_TYPE_OF(uint64_t, Uint64)
KMM_IMPL_DATA_TYPE_OF(float, Float32)
KMM_IMPL_DATA_TYPE_OF(double, Float64)
KMM_IMPL_DATA_TYPE_OF(KeyValue<int64_t>, KeyValueInt64)
KMM_IMPL_DATA_TYPE_OF(KeyValue<double>, KeyValueFloat64)

#undef KMM_IMPL_DATA_TYPE_OF

/// @}

}  // namespace kmm
