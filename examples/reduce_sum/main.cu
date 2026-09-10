// An example showing how to use `kmm::Accumulator` to compute the sum of a large vector by
// splitting it across every available GPU: each device reduces its own slice into a single value,
// and KMM combines the per-device results into one final sum.
#include <algorithm>
#if defined(KMM_USE_CUDA)
    #include <cuda_runtime.h>
#elif defined(KMM_USE_HIP)
    #include <hip/hip_runtime.h>
#endif
#include <iostream>
#include <vector>

#include "kmm/kmm.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/runtime.hpp"

static constexpr unsigned int BLOCK_SIZE = 256;
static constexpr unsigned int NUM_BLOCKS = 64;

__global__ void sum_kernel(kmm::View<float> input, kmm::ViewMut<float> output) {
    __shared__ float partial_sums[BLOCK_SIZE];

    auto tid = threadIdx.x;
    auto n = static_cast<unsigned int>(input.size());
    auto grid_stride = gridDim.x * BLOCK_SIZE;

    float sum = 0.0f;
    for (unsigned int i = blockIdx.x * BLOCK_SIZE + tid; i < n; i += grid_stride) {
        sum += input[i];
    }

    partial_sums[tid] = sum;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = partial_sums[0];
    }
}

int main() {
    auto config = kmm::default_config_from_environment();
    kmm::Context context = kmm::make_runtime(config);

    size_t n = 1'000'000;
    std::vector<float> input(n);

    for (size_t i = 0; i < n; i++) {
        input[i] = 1.0f;
    }

    auto values = context.from_vector(input);

    size_t num_devices = context.system_info().num_devices();
    size_t chunk_size = (n + num_devices - 1) / num_devices;

    auto accum = context.accumulator<float>(kmm::ReductionOp::Sum);

    for (size_t d = 0; d < num_devices; d++) {
        size_t begin = d * chunk_size;
        size_t end = std::min(begin + chunk_size, n);

        if (begin >= end) {
            break;
        }

        auto chunk = values.slice_axis<0>(begin, end);

        context.gpu(kmm::DeviceId(d))
            .submit(
                kmm::Kernel(sum_kernel, NUM_BLOCKS, BLOCK_SIZE),
                chunk,
                kmm::reduce(accum, NUM_BLOCKS)
            );
    }

    auto output = accum.finalize();
    float result = context.to_scalar(output);

    std::cout << "Sum of " << n << " ones across " << num_devices << " device(s): " << result
              << '\n';

    if (result != static_cast<float>(n)) {
        std::cerr << "unexpected result!\n";
        return 1;
    }

    return 0;
}
