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

/// A runtime tag for the scalar element type operated on by `reduce`.
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

/// The operator applied by `reduce`/`reduce_gpu` to combine elements.
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

/// Properties of a `DataType` tag, keyed on the tag itself. Each specialization provides:
///  - `element_type`: the C++ type stored in each element;
///  - `name`: a human-readable name (e.g. `"Float32"`).
template<DataType dtype>
struct data_type_traits;

/// The C++ element type corresponding to the `DataType` tag `dtype`.
template<DataType dtype>
using element_type_t = typename data_type_traits<dtype>::element_type;

/// Maps a C++ element type to its `DataType` tag via a `static constexpr DataType value` member.
template<typename T>
struct data_type_of_impl;

/// The `DataType` tag for `T`. Valid for every built-in element type and any type with a
/// `data_type_of_impl` specialization.
template<typename T>
constexpr DataType data_type_of() {
    return data_type_of_impl<T>::value;
}

#define KMM_IMPL_DATA_TYPE(TYPE, DTYPE)                    \
    template<>                                             \
    struct data_type_traits<DataType::DTYPE> {             \
        using element_type = TYPE;                         \
        static constexpr const char* name = #DTYPE;        \
    };                                                     \
    template<>                                             \
    struct data_type_of_impl<TYPE> {                       \
        static constexpr DataType value = DataType::DTYPE; \
    };

KMM_IMPL_DATA_TYPE(int32_t, Int32)
KMM_IMPL_DATA_TYPE(int64_t, Int64)
KMM_IMPL_DATA_TYPE(uint32_t, Uint32)
KMM_IMPL_DATA_TYPE(uint64_t, Uint64)
KMM_IMPL_DATA_TYPE(float, Float32)
KMM_IMPL_DATA_TYPE(double, Float64)
KMM_IMPL_DATA_TYPE(KeyValue<int64_t>, KeyValueInt64)
KMM_IMPL_DATA_TYPE(KeyValue<double>, KeyValueFloat64)

#undef KMM_IMPL_DATA_TYPE

/// @}

}  // namespace kmm
