#include <chrono>
#include <iostream>
#include <string>

#include "kmm/kmm.hpp"

using real_type = float;
const unsigned int max_iterations = 10;

__global__ void initialize_range(kmm::Range<int64_t> chunk, kmm::GPUSubviewMut<real_type> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin;
    if (i >= chunk.end) {
        return;
    }

    output[i] = static_cast<real_type>(i);
}

__global__ void fill_range(
    kmm::Range<int64_t> chunk,
    real_type value,
    kmm::GPUSubviewMut<real_type> output
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin;
    if (i >= chunk.end) {
        return;
    }

    output[i] = value;
}

__global__ void vector_add(
    kmm::Range<int64_t> range,
    kmm::GPUSubviewMut<real_type> output,
    kmm::GPUSubview<real_type> left,
    kmm::GPUSubview<real_type> right
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.begin;

    if (i >= range.end) {
        return;
    }

    output[i] = left[i] + right[i];
}

bool inner_loop(
    kmm::RuntimeHandle& rt,
    unsigned int threads,
    int64_t n,
    int64_t chunk_size,
    std::chrono::duration<double>& init_time,
    std::chrono::duration<double>& run_time
) {
    using namespace kmm::placeholders;

    dim3 block_size = threads;
    auto timing_start_init = std::chrono::steady_clock::now();
    auto A = kmm::Array<real_type> {n};
    auto B = kmm::Array<real_type> {n};
    auto C = kmm::Array<real_type> {n};

    // Initialize input arrays
    rt.parallel_submit(
        kmm::TileDomain(n, chunk_size),
        kmm::GPUKernel(initialize_range, block_size),
        _x,
        write(A[_x])
    );

    rt.parallel_submit(
        kmm::TileDomain(n, chunk_size),
        kmm::GPUKernel(fill_range, block_size),
        _x,
        static_cast<real_type>(1.0),
        write(B[_x])
    );

    rt.synchronize();
    auto timing_stop_init = std::chrono::steady_clock::now();
    init_time += timing_stop_init - timing_start_init;

    // Benchmark
    auto timing_start = std::chrono::steady_clock::now();
    rt.parallel_submit(
        kmm::TileDomain(n, chunk_size),
        kmm::GPUKernel(vector_add, block_size),
        _x,
        write(C[_x]),
        A[_x],
        B[_x]
    );

    rt.synchronize();
    auto timing_stop = std::chrono::steady_clock::now();
    run_time += timing_stop - timing_start;

    // Correctness check
    std::vector<real_type> result(n);
    C.copy_to(result);

    for (unsigned int i = 0; i < n; i++) {
        if (result[i] != static_cast<real_type>(i) + 1) {
            std::cerr << "Wrong result at " << i << " : " << result[i]
                      << " != " << static_cast<real_type>(i) + 1.0 << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    auto rt = kmm::make_runtime();
    bool status = false;
    int64_t n = 0;
    int64_t num_chunks = 0;
    unsigned int num_threads = 0;
    double ops = max_iterations;
    double mem = 3.0 * sizeof(real_type) * max_iterations;
    std::chrono::duration<double> init_time, vector_add_time;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <threads> <num_chunks> <size>" << std::endl;
        return 1;
    } else {
        num_threads = std::stoul(argv[1]);
        num_chunks = std::stoll(argv[2]);
        n = std::stoll(argv[3]);
    }
    ops *= double(n);
    mem *= double(n);

    // Warm-up run
    status = inner_loop(rt, num_threads, n, kmm::div_ceil(n, num_chunks), init_time, vector_add_time);
    if (!status) {
        std::cerr << "Warm-up run failed." << std::endl;
        return 1;
    }

    init_time = std::chrono::duration<double>();
    vector_add_time = std::chrono::duration<double>();

    for (unsigned int iteration = 0; iteration < max_iterations; ++iteration) {
        status = inner_loop(
            rt,
            num_threads,
            n,
            kmm::div_ceil(n, num_chunks),
            init_time,
            vector_add_time
        );
        if (!status) {
            std::cerr << "Run with " << num_chunks << " chunks failed." << std::endl;
            return 1;
        }
    }

    std::cout << "Performance with " << num_threads << " threads, " << num_chunks
              << " chunks, and n = " << n << std::endl;

    std::cout << "Total time (init): " << init_time.count() << " seconds" << std::endl;
    std::cout << "Average iteration time (init): " << init_time.count() / max_iterations
              << " seconds" << std::endl;

    std::cout << "Total time: " << vector_add_time.count() << " seconds" << std::endl;
    std::cout << "Average iteration time: " << vector_add_time.count() / max_iterations
              << " seconds" << std::endl;
    std::cout << "Throughput: " << (ops / vector_add_time.count()) / 1'000'000'000.0 << " GFLOP/s"
              << std::endl;
    std::cout << "Memory bandwidth: " << (mem / vector_add_time.count()) / 1'000'000'000.0
              << " GB/s" << std::endl;
    std::cout << std::endl;

    return 0;
}
