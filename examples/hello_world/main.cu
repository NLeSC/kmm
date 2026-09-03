// A minimal example showing the round trip of a buffer through KMM: a vector is written on the
// CPU, doubled on the GPU, and read back on the CPU.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "kmm/api/context.hpp"
#include "kmm/api/host.hpp"
#include "kmm/api/kernel.hpp"
#include "kmm/api/launch_arg.hpp"
#include "kmm/api/parallel_for.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/runtime.hpp"

__global__ void double_values(kmm::ViewMut<float> view) {
    auto i = static_cast<decltype(view.size())>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i < view.size()) {
        view[i] *= 2.0f;
    }
}

int main() {
    auto config = kmm::default_config_from_environment();
    config.host_memory_limit = 5ULL * 1024 * 1024 * 1024;
    kmm::Context context = kmm::make_runtime(config);
    auto host = context.host();
    auto dev = context.gpu();

    // Write a vector on the CPU.
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto array = host.from_vector(input);

    // Update the vector on the GPU: double every element.
    dev.access(write(array)).submit([](auto stream, auto view) {
        unsigned int block_size = 256;
        unsigned int grid_size =
            (static_cast<unsigned int>(view.size()) + block_size - 1) / block_size;

        double_values<<<grid_size, block_size, 0, stream>>>(view);
    });

    unsigned int block_size = 256;
    unsigned int grid_size =
        (static_cast<unsigned int>(array.size()) + block_size - 1) / block_size;

    host.prefetch(array, true);

    dev.submit(  //
        kmm::Kernel(double_values, grid_size, block_size),
        write(array)
    );

    host.prefetch(array, true);

    dev.parallel_for(
        array.shape(),
        KMM_LAMBDA(auto index, auto view) { view[index] *= 2.0f; },
        write(array)
    );

    // Read the vector back on the CPU.
    std::vector<float> output(input.size());

    {
        auto guard = host.access(array);
        const auto* ptr = guard.get().data();
        std::copy_n(ptr, input.size(), output.data());
    }

    std::cout << "Input:  ";
    for (float value : input) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    std::cout << "Output: ";
    for (float value : output) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    return 0;
}
