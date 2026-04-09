#include <float.h>

#include "gpu_operators.cuh"

#include "kmm/memops/gpu_fill.hpp"
#include "kmm/memops/gpu_reduction.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/gpu_utils.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

static constexpr size_t total_block_size = 256;

template<typename T, Reduction Op, bool UseAtomics = true, bool UseSmem = true>
__global__ void reduction_kernel(
    const T* src_buffer,
    T* dst_buffer,
    size_t num_outputs,
    size_t num_inputs_per_output,
    size_t input_stride,
    size_t items_per_thread
) {
    __shared__ T shared_results[total_block_size];

    uint32_t thread_x = threadIdx.x;
    uint32_t thread_y = threadIdx.y;

    uint64_t global_x = blockIdx.x * uint64_t(blockDim.x) + thread_x;
    uint64_t global_y = blockIdx.y * uint64_t(blockDim.y) * items_per_thread + thread_y;

    ReductionOperator<T, Op> reduce;
    T local_result = reduce.identity();

    if (global_x < num_outputs && global_y < num_inputs_per_output) {
        size_t x = global_x;
        size_t max_y = min(global_y + items_per_thread * blockDim.y, num_inputs_per_output);

        for (size_t y = global_y; y < max_y; y += blockDim.y) {
            T partial_result = src_buffer[y * input_stride + x];
            local_result = reduce(local_result, partial_result);
        }
    }

    if constexpr (UseSmem) {
        shared_results[thread_y * blockDim.x + thread_x] = local_result;

        __syncthreads();

        if (thread_y == 0) {
            for (unsigned int y = 1; y < blockDim.y; y++) {
                T partial_result = shared_results[y * blockDim.x + thread_x];
                local_result = reduce(local_result, partial_result);
            }
        }
    }

    if (global_x < num_outputs && thread_y == 0) {
        if constexpr (UseAtomics) {
            GPUAtomic<ReductionOperator<T, Op>>::atomic_combine(
                &dst_buffer[global_x],
                local_result
            );
        } else {
            dst_buffer[global_x] = local_result;
        }
    }
}

template<typename T, Reduction Op>
void execute_reduction_for_type_and_op(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    size_t num_outputs,
    size_t num_partials_per_output,
    size_t input_stride
) {
    size_t block_size_x;
    size_t block_size_y;
    size_t items_per_thread;

    if (num_partials_per_output <= 8) {
        block_size_x = total_block_size;
        block_size_y = 1;
        items_per_thread = num_partials_per_output;
    } else if (num_outputs < 32) {
        block_size_x = round_up_to_power_of_two(num_outputs);
        block_size_y = total_block_size / block_size_x;
        items_per_thread = 1;
    } else {
        block_size_x = 32;
        block_size_y = total_block_size / block_size_x;
        items_per_thread = 8;
    }

    // Divide the total number of elements by the total number of threads
    // We use max 512 blocks on the GPU as a rough heuristic here
    size_t max_blocks_per_gpu = 512;
    size_t max_grid_size_y = div_ceil(max_blocks_per_gpu, div_ceil(num_outputs, block_size_x));

    // If we do not have atomics, we can only have 1 block in the y-direction
    if (!IsGPUAtomicSupported<ReductionOperator<T, Op>>()) {
        max_grid_size_y = 1;
    }

    // The minimum items per thread is the number of partials divided by the maximum threads along Y
    size_t min_items_per_thread = div_ceil(num_partials_per_output, max_grid_size_y * block_size_y);

    if (items_per_thread < min_items_per_thread) {
        items_per_thread = min_items_per_thread;
    }

    dim3 block_size = {
        checked_cast<unsigned int>(block_size_x),
        checked_cast<unsigned int>(block_size_y),
    };

    dim3 grid_size = {
        checked_cast<unsigned int>(div_ceil(num_outputs, block_size_x)),
        checked_cast<unsigned int>(
            div_ceil(num_partials_per_output, block_size_y * items_per_thread)
        )
    };

    if constexpr (IsReductionSupported<T, Op>()) {
        if (grid_size.y == 1 && block_size_y == 1) {
            reduction_kernel<T, Op, false, false><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const T*>(src_buffer),
                reinterpret_cast<T*>(dst_buffer),
                num_outputs,
                num_partials_per_output,
                input_stride,
                items_per_thread
            );

            KMM_GPU_CHECK(gpu_get_last_error());
            return;
        }

        if (grid_size.y == 1) {
            reduction_kernel<T, Op, false><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const T*>(src_buffer),
                reinterpret_cast<T*>(dst_buffer),
                num_outputs,
                num_partials_per_output,
                input_stride,
                items_per_thread
            );

            KMM_GPU_CHECK(gpu_get_last_error());
            return;
        }

        if constexpr (IsGPUAtomicSupported<ReductionOperator<T, Op>>()) {
            T identity = ReductionOperator<T, Op>::identity();
            execute_gpu_fill_async(stream, dst_buffer, FillDef::with_value(identity, num_outputs));

            reduction_kernel<T, Op><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const T*>(src_buffer),
                reinterpret_cast<T*>(dst_buffer),
                num_outputs,
                num_partials_per_output,
                input_stride,
                items_per_thread
            );

            KMM_GPU_CHECK(gpu_get_last_error());
            return;
        }
    }

    // silence unused warnings
    (void)input_stride;
    (void)stream;
    (void)src_buffer;
    (void)dst_buffer;
    (void)block_size;

    throw std::runtime_error(
        fmt::format("reduction {} for data type {} is not yet supported", Op, DataType::of<T>())
    );
}

template<typename T>
void execute_reduction_for_type(
    g_stream_t stream,
    Reduction operation,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    size_t num_outputs,
    size_t num_partials_per_output,
    size_t input_stride
) {
#define KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(O) \
    execute_reduction_for_type_and_op<T, O>(  \
        stream,                               \
        src_buffer,                           \
        dst_buffer,                           \
        num_outputs,                          \
        num_partials_per_output,              \
        input_stride                          \
    );

    switch (operation) {
        case Reduction::Sum:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(Reduction::Sum)
            break;
        case Reduction::Product:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(Reduction::Product)
            break;
        case Reduction::Min:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(Reduction::Min)
            break;
        case Reduction::Max:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(Reduction::Max)
            break;
        case Reduction::BitAnd:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(Reduction::BitAnd)
            break;
        case Reduction::BitOr:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(Reduction::BitOr)
            break;
        default:
            throw std::runtime_error(
                fmt::format("reductions for operation {} are not yet supported", operation)
            );
    }
}

void execute_gpu_reduction_async(
    g_stream_t stream,
    g_device_ptr_t src_buffer,
    g_device_ptr_t dst_buffer,
    ReductionDef reduction
) {
#define KMM_CALL_REDUCTION_FOR_TYPE(T)                                    \
    execute_reduction_for_type<T>(                                        \
        stream,                                                           \
        reduction.operation,                                              \
        (g_device_ptr_t)((unsigned long long)(src_buffer)                 \
                         + reduction.input_offset_elements * sizeof(T)),  \
        (g_device_ptr_t)((unsigned long long)(dst_buffer)                 \
                         + reduction.output_offset_elements * sizeof(T)), \
        reduction.num_outputs,                                            \
        reduction.num_inputs_per_output,                                  \
        reduction.input_stride_elements                                   \
    );

#define KMM_CALL_REDUCTION_FOR_COMPLEX(T)                                     \
    execute_reduction_for_type<T>(                                            \
        stream,                                                               \
        reduction.operation,                                                  \
        (g_device_ptr_t)((unsigned long long)(src_buffer)                     \
                         + reduction.input_offset_elements * 2 * sizeof(T)),  \
        (g_device_ptr_t)((unsigned long long)(dst_buffer)                     \
                         + reduction.output_offset_elements * 2 * sizeof(T)), \
        2 * reduction.num_outputs,                                            \
        reduction.num_inputs_per_output,                                      \
        2 * reduction.input_stride_elements                                   \
    );

    switch (reduction.data_type.as_scalar()) {
        case ScalarType::Int8:
            KMM_CALL_REDUCTION_FOR_TYPE(int8_t)
            return;
        case ScalarType::Int16:
            KMM_CALL_REDUCTION_FOR_TYPE(int16_t)
            return;
        case ScalarType::Int32:
            KMM_CALL_REDUCTION_FOR_TYPE(int32_t)
            return;
        case ScalarType::Int64:
            KMM_CALL_REDUCTION_FOR_TYPE(int64_t)
            return;
        case ScalarType::Uint8:
            KMM_CALL_REDUCTION_FOR_TYPE(uint8_t)
            return;
        case ScalarType::Uint16:
            KMM_CALL_REDUCTION_FOR_TYPE(uint16_t)
            return;
        case ScalarType::Uint32:
            KMM_CALL_REDUCTION_FOR_TYPE(uint32_t)
            return;
        case ScalarType::Uint64:
            KMM_CALL_REDUCTION_FOR_TYPE(uint64_t)
            return;
        case ScalarType::Float32:
            KMM_CALL_REDUCTION_FOR_TYPE(float)
            return;
        case ScalarType::Float64:
            KMM_CALL_REDUCTION_FOR_TYPE(double)
            return;
        case ScalarType::KeyAndInt64:
            KMM_CALL_REDUCTION_FOR_TYPE(KeyValue<int64_t>)
            return;
        case ScalarType::KeyAndFloat64:
            KMM_CALL_REDUCTION_FOR_TYPE(KeyValue<double>)
            return;
        case ScalarType::Complex32:
            KMM_CALL_REDUCTION_FOR_COMPLEX(float)
            return;
        case ScalarType::Complex64:
            KMM_CALL_REDUCTION_FOR_COMPLEX(double)
            return;
        default:
            throw std::runtime_error(
                fmt::format("reductions on data type {} are not yet supported", reduction.data_type)
            );
    }
}
}  // namespace kmm