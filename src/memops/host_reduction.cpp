#include "host_operators.hpp"

#include "kmm/memops/host_reduction.hpp"

namespace kmm {

template<size_t NumRows, typename T, Reduction Op>
KMM_NOINLINE void execute_reduction_few_rows(
    const T* __restrict__ src_buffer,
    T* __restrict__ dst_buffer,
    size_t num_columns,
    size_t row_stride
) {
    for (size_t j = 0; j < num_columns; j++) {
        dst_buffer[j] = src_buffer[j];

        for (size_t i = 1; i < NumRows; i++) {
            dst_buffer[j] =
                ReductionOperator<T, Op>()(dst_buffer[j], src_buffer[(i * row_stride) + j]);
        }
    }
}

template<typename T, Reduction Op>
KMM_NOINLINE void execute_reduction_basic(
    const T* __restrict__ src_buffer,
    T* __restrict__ dst_buffer,
    size_t num_columns,
    size_t num_rows,
    size_t row_stride
) {
    for (size_t j = 0; j < num_columns; j++) {
        dst_buffer[j] = src_buffer[j];
    }

    for (size_t i = 1; i < num_rows; i++) {
        for (size_t j = 0; j < num_columns; j++) {
            dst_buffer[j] =
                ReductionOperator<T, Op>()(dst_buffer[j], src_buffer[(i * row_stride) + j]);
        }
    }
}

template<typename T, Reduction Op>
KMM_NOINLINE void execute_reduction_impl(
    const T* src_buffer,
    T* dst_buffer,
    size_t num_columns,
    size_t num_rows,
    size_t row_stride
) {
    // For zero rows, we just fill with the identity value.
    if (num_rows == 0) {
        std::fill_n(dst_buffer, num_columns, ReductionOperator<T, Op>::identity());
        return;
    }

#define KMM_IMPL_REDUCTION_CASE(N)                     \
    if (num_rows == (N)) {                             \
        return execute_reduction_few_rows<(N), T, Op>( \
            src_buffer,                                \
            dst_buffer,                                \
            num_columns,                               \
            row_stride                                 \
        );                                             \
    }

    // Specialize based on the number of rows
    KMM_IMPL_REDUCTION_CASE(1)
    KMM_IMPL_REDUCTION_CASE(2)
    KMM_IMPL_REDUCTION_CASE(3)
    KMM_IMPL_REDUCTION_CASE(4)
    KMM_IMPL_REDUCTION_CASE(5)
    KMM_IMPL_REDUCTION_CASE(6)
    KMM_IMPL_REDUCTION_CASE(7)
    KMM_IMPL_REDUCTION_CASE(8)

    return execute_reduction_basic<T, Op>(
        src_buffer,
        dst_buffer,
        num_columns,
        num_rows,
        row_stride
    );
}

void execute_reduction(const void* src_buffer, void* dst_buffer, ReductionDef reduction) {
#define KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, OP)                                              \
    if constexpr (IsReductionSupported<T, Reduction::OP>()) {                                  \
        if (reduction.operation == Reduction::OP) {                                            \
            execute_reduction_impl<                                                            \
                T,                                                                             \
                Reduction::OP>(/* NOLINTNEXTLINE */                                            \
                               static_cast<const T*>(src_buffer)                               \
                                   + reduction.input_offset_elements, /* NOLINTNEXTLINE */     \
                               static_cast<T*>(dst_buffer) + reduction.output_offset_elements, \
                               reduction.num_outputs,                                          \
                               reduction.num_inputs_per_output,                                \
                               reduction.input_stride_elements                                 \
            );                                                                                 \
            return;                                                                            \
        }                                                                                      \
    }

#define KMM_CALL_REDUCTION_FOR_TYPE(T)             \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Sum)     \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Product) \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Min)     \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Max)     \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, BitAnd)  \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, BitOr)

    switch (reduction.data_type.as_scalar()) {
        case ScalarType::Int8:
            KMM_CALL_REDUCTION_FOR_TYPE(int8_t)
            break;
        case ScalarType::Int16:
            KMM_CALL_REDUCTION_FOR_TYPE(int16_t)
            break;
        case ScalarType::Int32:
            KMM_CALL_REDUCTION_FOR_TYPE(int32_t)
            break;
        case ScalarType::Int64:
            KMM_CALL_REDUCTION_FOR_TYPE(int64_t)
            break;
        case ScalarType::Uint8:
            KMM_CALL_REDUCTION_FOR_TYPE(uint8_t)
            break;
        case ScalarType::Uint16:
            KMM_CALL_REDUCTION_FOR_TYPE(uint16_t)
            break;
        case ScalarType::Uint32:
            KMM_CALL_REDUCTION_FOR_TYPE(uint32_t)
            break;
        case ScalarType::Uint64:
            KMM_CALL_REDUCTION_FOR_TYPE(uint64_t)
            break;
        case ScalarType::Float32:
            KMM_CALL_REDUCTION_FOR_TYPE(float)
            break;
        case ScalarType::Float64:
            KMM_CALL_REDUCTION_FOR_TYPE(double)
            break;
        case ScalarType::Complex32:
            KMM_CALL_REDUCTION_FOR_TYPE(std::complex<float>)
            break;
        case ScalarType::Complex64:
            KMM_CALL_REDUCTION_FOR_TYPE(std::complex<double>)
            break;
        case ScalarType::KeyAndInt64:
            KMM_CALL_REDUCTION_FOR_TYPE(KeyValue<int64_t>)
            break;
        case ScalarType::KeyAndFloat64:
            KMM_CALL_REDUCTION_FOR_TYPE(KeyValue<double>)
            break;
        default:
            break;
    }

    throw std::runtime_error("unsupported reduction operation");
}

}  // namespace kmm