![KMM: Kernel Memory Manager](https://raw.githubusercontent.com/NLeSC/kmm/refs/heads/main/docs/_static/kmm-logo.png)


#
[![CPU Build Status](https://github.com/NLeSC/kmm/actions/workflows/cmake-multi-compiler.yml/badge.svg)](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-multi-compiler.yml)
[![CUDA Build Status](https://github.com/NLeSC/kmm/actions/workflows/cmake-cuda-multi-compiler.yml/badge.svg)](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-cuda-multi-compiler.yml)
[![HIP Build Status](https://github.com/NLeSC/kmm/actions/workflows/cmake-hip.yml/badge.svg)](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-hip.yml)

The **Kernel Memory Manager** (KMM) is a lightweight, high-performance framework designed for parallel dataflow execution and efficient memory management on multi-GPU platforms.


KMM automatically manages GPU memory, partitions workloads across multiple GPUs, and schedules tasks efficiently.
Unlike frameworks that require a specific programming model, KMM integrates existing GPU kernels or functions without the need to fully rewrite your code.


## Features

* **Efficient Memory Management**: automatically allocates memory and transfers data between GPU and host only when neccessary.
* **Scalable Computing**: seamlessly spills data from GPU to host memory, enabling huge datasets that exceed GPU memory.
* **Optimized Scheduling**: DAG scheduler automatically tracks dependencies and executes kernels in a sequentially consistent order.
* **Flexible Work Partitioning**: Split workloads and data according to user-defined distributions, ensuring utilization of available resources.
* **Portable Execution**: supports existing CUDA, HIP, and CPU-based functions; seamless integration with minimal changes.
* **Multi-Dimensional Arrays**: handles ND-arrays of any shape, dimensionality, and data type.


## Resources

* [Full documentation](https://nlesc.github.io/kmm)


## Example

Example: A simple vector add kernel:

```C++
#include "kmm/kmm.hpp"

__global__ void vector_add(
    kmm::Range<int64_t> range,
    kmm::GPUSubviewMut<float> output,
    kmm::GPUSubview<float> left,
    kmm::GPUSubview<float> right
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.begin;
    if (i >= range.end) return;

    output[i] = left[i] + right[i];
}

int main() {
    // 2B items, 10 chunks, 256 threads per block
    long n = 2'000'000'000;
    long chunk_size = n / 10;
    dim3 block_size = 256;

    // Initialize runtime
    auto rt = kmm::make_runtime();

    // Create arrays
    auto A = kmm::Array<float> {n};
    auto B = kmm::Array<float> {n};
    auto C = kmm::Array<float> {n};

    // Initialize input arrays
    initialize_inputs(A, B);

    // Launch the kernel!
    rt.parallel_submit(
        n, chunk_size,
        kmm::GPUKernel(vector_add, block_size),
        _x,
        write(C[_x]),
        A[_x],
        B[_x]
    );

    // Wait for completion
    rt.synchronize();

    return 0;
}
```


## License

KMM is made available under the terms of the Apache License version 2.0, see the file LICENSE for details.

