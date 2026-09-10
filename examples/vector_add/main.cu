// A minimal example that adds two vectors on the GPU using `Device::parallel_for`: `c[i] = a[i] +
// b[i]` for every index `i`, with KMM taking care of allocating and moving the buffers.
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

int main() {
    auto config = kmm::default_config_from_environment();
    kmm::Context context = kmm::make_runtime(config);
    auto host = context.host();
    auto dev = context.gpu();

    size_t n = 1'000'000;
    std::vector<float> input_a(n);
    std::vector<float> input_b(n);

    for (size_t i = 0; i < n; i++) {
        input_a[i] = static_cast<float>(i);
        input_b[i] = static_cast<float>(2 * i);
    }

    // Move the input vectors to the runtime and allocate an (uninitialized) output vector.
    auto a = host.from_vector(input_a);
    auto b = host.from_vector(input_b);
    auto c = context.empty<float>(n);

    // Launch one GPU thread per element: `c[index] = a[index] + b[index]`.
    dev.parallel_for(
        c.shape(),
        KMM_LAMBDA(auto index, auto va, auto vb, auto vc) { vc[index] = va[index] + vb[index]; },
        a,
        b,
        write(c)
    );

    std::vector<float> output = context.to_vector(c);

    // Verify (and print) the first few results.
    for (size_t i = 0; i < 10; i++) {
        std::cout << input_a[i] << " + " << input_b[i] << " = " << output[i] << '\n';
    }

    for (size_t i = 0; i < n; i++) {
        if (output[i] != input_a[i] + input_b[i]) {
            std::cerr << "mismatch at index " << i << '\n';
            return 1;
        }
    }

    std::cout << "All " << n << " elements match!\n";
    return 0;
}
