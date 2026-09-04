#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "kmm/kmm.hpp"

using real_type = float;
const unsigned int max_iterations = 10;

__global__ void initialize_range(int64_t offset, kmm::ViewMut<real_type> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output.size()) {
        return;
    }

    output[i] = static_cast<real_type>(offset + i);
}

__global__ void fill_range(real_type value, kmm::ViewMut<real_type> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output.size()) {
        return;
    }

    output[i] = value;
}

__global__ void vector_add(
    kmm::ViewMut<real_type> output,
    kmm::View<real_type> left,
    kmm::View<real_type> right
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output.size()) {
        return;
    }

    output[i] = left[i] + right[i];
}

// Launches `kernel` (previously bound to a fixed block size) over `extent` elements, computing
// the grid size to cover it.
template<typename K, typename... Args>
void launch(
    kmm::Device& device,
    K kernel,
    unsigned int block_size,
    int64_t extent,
    Args&&... args
) {
    auto grid_size = static_cast<unsigned int>(kmm::div_ceil(extent, int64_t(block_size)));

    device.submit(kmm::Kernel(kernel, dim3(grid_size), dim3(block_size)), std::forward<Args>(args)...);
}

bool inner_loop(
    kmm::Context& context,
    std::vector<kmm::Device>& devices,
    unsigned int threads,
    int64_t n,
    int64_t chunk_size,
    std::chrono::duration<double>& init_time,
    std::chrono::duration<double>& run_time
) {
    auto timing_start_init = std::chrono::steady_clock::now();

    // Each chunk of `A`/`B`/`C` is its own array (own buffer, own home device), which is what
    // lets `num_chunks` independent submissions below actually run concurrently/pipelined,
    // rather than serializing on one shared buffer.
    kmm::Distribution<1> dist{kmm::Shape<1>(n), kmm::Shape<1>(chunk_size)};
    kmm::DistArray<real_type, 1> A(context.runtime(), dist);
    kmm::DistArray<real_type, 1> B(context.runtime(), dist);
    kmm::DistArray<real_type, 1> C(context.runtime(), dist);

    // Initialize input arrays
    for (size_t i = 0; i < dist.num_chunks(); i++) {
        auto& device = devices[A.chunk_home(i).as_device().get()];
        int64_t offset = dist.chunk_offset(dist.unravel(i))[0];
        int64_t extent = A.chunk(i).size();

        launch(device, initialize_range, threads, extent, offset, kmm::write(A.chunk(i)));
        launch(device, fill_range, threads, extent, static_cast<real_type>(1.0), kmm::write(B.chunk(i)));
    }

    context.synchronize();
    auto timing_stop_init = std::chrono::steady_clock::now();
    init_time += timing_stop_init - timing_start_init;

    // Benchmark
    auto timing_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < dist.num_chunks(); i++) {
        auto& device = devices[C.chunk_home(i).as_device().get()];
        int64_t extent = C.chunk(i).size();

        launch(device, vector_add, threads, extent, kmm::write(C.chunk(i)), A.chunk(i), B.chunk(i));
    }

    context.synchronize();
    auto timing_stop = std::chrono::steady_clock::now();
    run_time += timing_stop - timing_start;

    // Correctness check
    bool status = true;

    for (size_t i = 0; i < dist.num_chunks(); i++) {
        int64_t offset = dist.chunk_offset(dist.unravel(i))[0];
        std::vector<real_type> result = context.to_vector(C.chunk(i));

        for (size_t j = 0; j < result.size(); j++) {
            int64_t global_i = offset + static_cast<int64_t>(j);
            auto expected = static_cast<real_type>(global_i) + static_cast<real_type>(1.0);

            if (result[j] != expected) {
                std::cerr << "Wrong result at " << global_i << " : " << result[j]
                          << " != " << expected << std::endl;
                status = false;
            }
        }
    }

    return status;
}

int main(int argc, char* argv[]) {
    auto config = kmm::default_config_from_environment();
    kmm::Context context = kmm::make_runtime(config);

    // One `Device` per physical GPU, created once and reused across chunks/iterations --
    // constructing a `Device` owns a GPU stream, so it should not happen inside the hot loop.
    size_t num_devices = context.system_info().num_devices();
    std::vector<kmm::Device> devices;
    for (size_t i = 0; i < num_devices; i++) {
        devices.push_back(context.gpu(kmm::DeviceId(i)));
    }

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
    status = inner_loop(
        context,
        devices,
        num_threads,
        n,
        kmm::div_ceil(n, num_chunks),
        init_time,
        vector_add_time
    );
    if (!status) {
        std::cerr << "Warm-up run failed." << std::endl;
        return 1;
    }

    init_time = std::chrono::duration<double>();
    vector_add_time = std::chrono::duration<double>();

    for (unsigned int iteration = 0; iteration < max_iterations; ++iteration) {
        status = inner_loop(
            context,
            devices,
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
