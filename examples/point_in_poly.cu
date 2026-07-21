#ifdef KMM_USE_CUDA
    #include <curand_kernel.h>
#elif KMM_USE_HIP
    #include <rocrand/rocrand_kernel.h>
#endif

#include "spdlog/spdlog.h"

#include "kmm/api/launcher.hpp"
#include "kmm/api/mapper.hpp"
#include "kmm/api/runtime_handle.hpp"

__global__ void cn_pnpoly(
    kmm::Range<int> chunk,
    kmm::GPUSubviewMut<int> bitmap,
    kmm::GPUSubview<float2> points,
    int nvertices,
    kmm::GPUView<float2> vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin;

    if (i < chunk.end) {
        int c = 0;
        float2 p = points[i];

        int k = nvertices - 1;

        for (int j = 0; j < nvertices; k = j++) {  // edge from v to vp
            float2 vj = vertices[j];
            float2 vk = vertices[k];

            float slope = (vk.x - vj.x) / (vk.y - vj.y);

            if (((vj.y > p.y) != (vk.y > p.y)) &&  //if p is between vj and vk vertically
                (p.x < slope * (p.y - vj.y) + vj.x
                )) {  //if p.x crosses the line vj-vk when moved in positive x-direction
                c = !c;
            }
        }

        bitmap[i] = c;  // 0 if even (out), and 1 if odd (in)
    }
}

__global__ void init_points(kmm::Range<int> chunk, kmm::GPUSubviewMut<float2> points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin;

    if (i < chunk.end) {
#if __CUDA_ARCH__
        curandStatePhilox4_32_10_t state;
        curand_init(1234, i, 0, &state);
        points[i] = {curand_normal(&state), curand_normal(&state)};
#elif __HIP_DEVICE_COMPILE__
        rocrand_state_philox4x32_10 state;
        rocrand_init(1234, i, 0, &state);
        points[i] = {rocrand_normal(&state), rocrand_normal(&state)};
#endif
    }
}

void init_polygon(kmm::Range<int> chunk, int nvertices, kmm::ViewMut<float2> vertices) {
    for (int64_t i = chunk.begin; i < chunk.end; i++) {
        float angle = float(i) / float(nvertices) * float(2.0F * M_PI);
        vertices[i] = {cosf(angle), sinf(angle)};
    }
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int nvertices = 1000;
    int npoints = 1'000'000'000;
    int npoints_per_chunk = npoints / 10;
    dim3 block_size = 256;

    auto vertices = kmm::Array<float2> {nvertices};
    auto points = kmm::Array<float2> {npoints};
    auto bitmap = kmm::Array<int> {npoints};

    rt.submit(
        kmm::ResourceId::host(),
        kmm::Host(init_polygon),
        kmm::Range(nvertices),
        nvertices,
        write(vertices)
    );

    rt.parallel_submit(
        {npoints},
        {npoints_per_chunk},
        kmm::GPUKernel(init_points, block_size),
        _x,
        write(points[_x])
    );

    rt.parallel_submit(
        {npoints},
        {npoints_per_chunk},
        kmm::GPUKernel(cn_pnpoly, block_size),
        _x,
        write(bitmap[_x]),
        points[_x],
        nvertices,
        vertices
    );

    rt.synchronize();

    return EXIT_SUCCESS;
}
