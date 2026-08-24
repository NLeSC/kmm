#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <limits>
#include <string>

#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/utils/gpu_utils.hpp"

#define KMM_CUB_CHECK(...)                                                                      \
    do {                                                                                        \
        auto __cub_code = (__VA_ARGS__);                                            \
        if (KMM_UNLIKELY(__cub_code != cudaSuccess)) {                                  \
            throw ::kmm::GPUException(                                                          \
                std::string("GPU runtime error at ") + __FILE__ + ":" + std::to_string(__LINE__) \
                + " (" #__VA_ARGS__ "): " + gpuGetErrorStringRuntime(__cub_code)                 \
            );                                                                                   \
        }                                                                                        \
    } while (0)

namespace kmm {

namespace {

constexpr int REDUCE_THREADS_PER_BLOCK = 128;

// Per-axis extent plus the input/output strides, converted from bytes to element counts. Passed
// by value into device-side functors/kernels, so it must stay a small, trivially-copyable POD.
struct AxisLayout {
    int extent = 1;
    long long input_stride_elems = 0;
    long long output_stride_elems = 0;
};

struct BatchLayout {
    int num_dims = 0;
    AxisLayout dims[MEMOPS_MAX_DIMS] = {};
};

// Decomposes a linear output index into per-axis batch indices (row-major: the last axis in
// `dims` varies fastest) and sums the corresponding input/output element offset. Shared by the
// CUB offset functors (input side) and the scatter kernel (output side).
KMM_HOST_DEVICE
long long decompose_input_offset(const BatchLayout& layout, int linear_index) {
    long long offset = 0;
    int remaining = linear_index;

    for (int axis = layout.num_dims - 1; axis >= 0; axis--) {
        int idx = remaining % layout.dims[axis].extent;
        remaining /= layout.dims[axis].extent;
        offset += static_cast<long long>(idx) * layout.dims[axis].input_stride_elems;
    }

    return offset;
}

KMM_HOST_DEVICE
long long decompose_output_offset(const BatchLayout& layout, int linear_index) {
    long long offset = 0;
    int remaining = linear_index;

    for (int axis = layout.num_dims - 1; axis >= 0; axis--) {
        int idx = remaining % layout.dims[axis].extent;
        remaining /= layout.dims[axis].extent;
        offset += static_cast<long long>(idx) * layout.dims[axis].output_stride_elems;
    }

    return offset;
}

// Maps a segment index to its first input element, in units of `T` relative to `src_addr`
// (recall the reduction axis itself is required to be contiguous, so a segment is simply
// `reduction_extent` consecutive elements starting here).
struct BeginOffsetFunctor {
    BatchLayout layout;

    KMM_HOST_DEVICE
    long long operator()(int segment) const {
        return decompose_input_offset(layout, segment);
    }
};

struct EndOffsetFunctor {
    BatchLayout layout;
    long long reduction_extent;

    KMM_HOST_DEVICE
    long long operator()(int segment) const {
        return decompose_input_offset(layout, segment) + reduction_extent;
    }
};

// `cub::Sum`/`cub::Min`/`cub::Max` are built in; CUB has no `Product`.
template<typename T>
struct ProductOp {
    KMM_HOST_DEVICE
    T operator()(const T& a, const T& b) const {
        return static_cast<T>(a * b);
    }
};

// CUB always reduces into a plain contiguous buffer (`d_scratch`, one element per output). This
// kernel scatters those results into the real (possibly strided) output buffer, combining with
// the existing value when `accumulate` is set.
template<typename T, typename ReduceOp>
__global__ void scatter_reduce_results(
    const T* scratch,
    T* dst,
    BatchLayout layout,
    int num_outputs,
    bool accumulate,
    ReduceOp op
) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= num_outputs) {
        return;
    }

    long long offset = decompose_output_offset(layout, i);
    T value = scratch[i];

    dst[offset] = accumulate ? op(dst[offset], value) : value;
}

template<typename T>
T identity_for(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
            return static_cast<T>(0);
        case ReductionOp::Product:
            return static_cast<T>(1);
        case ReductionOp::Min:
            return std::numeric_limits<T>::max();
        case ReductionOp::Max:
            return std::numeric_limits<T>::lowest();
    }

    KMM_PANIC("invalid reduction operator");
}

template<typename T, typename ReduceOp>
void run_reduce(
    GPUStream stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description,
    ReduceOp op,
    T identity
) {
    src_addr = static_cast<const std::byte*>(src_addr) + description.input_offset;
    dst_addr = static_cast<std::byte*>(dst_addr) + description.output_offset;

    auto element_size = static_cast<memops_stride_type>(sizeof(T));

    // CUB's segmented reduce only understands contiguous segments, so the reduction axis must be
    // contiguous, and every batch stride must land on whole elements. Anything else (e.g. a
    // strided reduction axis) is not supported by this implementation.
    bool supported = description.reduction_stride == element_size;

    for (size_t i = 0; supported && i < description.num_dims; i++) {
        supported = supported && description.dims[i].input_stride % element_size == 0
            && description.dims[i].output_stride % element_size == 0;
    }

    if (!supported) {
        KMM_PANIC(
            "reduce_async: unsupported layout (the reduction axis must be contiguous, and every "
            "batch axis stride must be a multiple of the element size)"
        );
    }

    BatchLayout layout;
    layout.num_dims = static_cast<int>(description.num_dims);

    for (size_t i = 0; i < description.num_dims; i++) {
        layout.dims[i].extent = static_cast<int>(description.dims[i].extent);
        layout.dims[i].input_stride_elems = description.dims[i].input_stride / element_size;
        layout.dims[i].output_stride_elems = description.dims[i].output_stride / element_size;
    }

    int num_outputs = static_cast<int>(description.num_outputs());
    auto reduction_extent = static_cast<long long>(description.reduction_extent);
    bool accumulate = description.accumulate;

    GPUDeviceptr scratch_ptr = 0;
    KMM_GPU_CHECK(
        gpuMemAllocAsync(&scratch_ptr, sizeof(T) * static_cast<size_t>(num_outputs), stream)
    );
    T* d_scratch = reinterpret_cast<T*>(scratch_ptr);
    const T* d_in = static_cast<const T*>(src_addr);

    cub::CountingInputIterator<int> counting(0);
    cub::TransformInputIterator<long long, BeginOffsetFunctor, cub::CountingInputIterator<int>>
        begin_offsets(counting, BeginOffsetFunctor {layout});
    cub::TransformInputIterator<long long, EndOffsetFunctor, cub::CountingInputIterator<int>>
        end_offsets(counting, EndOffsetFunctor {layout, reduction_extent});

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First call (with `d_temp_storage == nullptr`) only computes `temp_storage_bytes`.
    KMM_CUB_CHECK(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_scratch,
        num_outputs,
        begin_offsets,
        end_offsets,
        op,
        identity,
        stream
    ));

    GPUDeviceptr temp_ptr = 0;

    if (temp_storage_bytes > 0) {
        KMM_GPU_CHECK(gpuMemAllocAsync(&temp_ptr, temp_storage_bytes, stream));
        d_temp_storage = reinterpret_cast<void*>(temp_ptr);
    }

    KMM_CUB_CHECK(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_scratch,
        num_outputs,
        begin_offsets,
        end_offsets,
        op,
        identity,
        stream
    ));

    if (temp_ptr != 0) {
        KMM_GPU_CHECK(gpuMemFreeAsync(temp_ptr, stream));
    }

    int blocks = (num_outputs + REDUCE_THREADS_PER_BLOCK - 1) / REDUCE_THREADS_PER_BLOCK;
    scatter_reduce_results<T, ReduceOp><<<blocks, REDUCE_THREADS_PER_BLOCK, 0, stream>>>(
        d_scratch,
        static_cast<T*>(dst_addr),
        layout,
        num_outputs,
        accumulate,
        op
    );
    KMM_CUB_CHECK(gpuGetLastError());

    KMM_GPU_CHECK(gpuMemFreeAsync(scratch_ptr, stream));
}

template<typename T>
void reduce_typed_async(
    GPUStream stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    switch (description.operation) {
        case ReductionOp::Sum:
            return run_reduce<T>(
                stream,
                src_addr,
                dst_addr,
                description,
                cub::Sum {},
                identity_for<T>(ReductionOp::Sum)
            );
        case ReductionOp::Product:
            return run_reduce<T>(
                stream,
                src_addr,
                dst_addr,
                description,
                ProductOp<T> {},
                identity_for<T>(ReductionOp::Product)
            );
        case ReductionOp::Min:
            return run_reduce<T>(
                stream,
                src_addr,
                dst_addr,
                description,
                cub::Min {},
                identity_for<T>(ReductionOp::Min)
            );
        case ReductionOp::Max:
            return run_reduce<T>(
                stream,
                src_addr,
                dst_addr,
                description,
                cub::Max {},
                identity_for<T>(ReductionOp::Max)
            );
    }

    KMM_PANIC("invalid reduction operator");
}

}  // namespace

void reduce_async(
    GPUStream stream,
    const void* src_addr,
    void* dst_addr,
    const ReductionDescription& description
) {
    if (description.is_equivalent_to_copy()) {
        return copy_async(stream, src_addr, dst_addr, description.as_copy());
    }

    switch (description.dtype) {
        case DataType::Int8:
            return reduce_typed_async<int8_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Int16:
            return reduce_typed_async<int16_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Int32:
            return reduce_typed_async<int32_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Int64:
            return reduce_typed_async<int64_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Uint8:
            return reduce_typed_async<uint8_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Uint16:
            return reduce_typed_async<uint16_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Uint32:
            return reduce_typed_async<uint32_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Uint64:
            return reduce_typed_async<uint64_t>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Float32:
            return reduce_typed_async<float>(
                stream,
                src_addr,
                dst_addr,
                description
            );
        case DataType::Float64:
            return reduce_typed_async<double>(
                stream,
                src_addr,
                dst_addr,
                description
            );
    }

    KMM_PANIC("invalid data type");
}

}  // namespace kmm
