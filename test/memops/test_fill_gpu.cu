#include <algorithm>
#include <cstddef>
#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/fill.hpp"
#include "kmm/runtime/memops/fill_gpu.hpp"
#include "kmm/utils/gpu_utils.hpp"

using namespace kmm;
using namespace kmm::memops;

static void test_fill(const FillDescription& desc, std::byte* data, size_t axis = 0) {
    if (axis >= desc.num_dims) {
        std::copy_n(desc.value.buffer, desc.value.length, &data[desc.offset]);
    } else {
        for (memops_extent_type i = 0; i < desc.dims[axis].extent; i++) {
            test_fill(desc, data + desc.dims[axis].stride * i, axis + 1);
        }
    }
}

static size_t size_fill(const FillDescription& desc, size_t offset = 0, size_t axis = 0) {
    if (axis >= desc.num_dims) {
        offset += desc.offset + desc.value.length;
    } else {
        for (memops_extent_type i = 0; i < desc.dims[axis].extent; i++) {
            offset =
                std::max(offset, size_fill(desc, offset + desc.dims[axis].stride * i, axis + 1));
        }
    }

    return offset;
}

bool check_fill_gpu(const FillDescription& desc) {
    g_device_t device = 0;
    g_context_t context = nullptr;
    g_stream_t stream = nullptr;

    KMM_GPU_CHECK(g_init(0));
    KMM_GPU_CHECK(g_device_get(&device, 0));
    KMM_GPU_CHECK(g_device_primary_ctx_retain(&context, device));
    KMM_GPU_CHECK(g_ctx_push_current(context));

    size_t size = size_fill(desc) * 2;  // *2 just to be sure

    std::vector<std::byte> reference(size);
    test_fill(desc, reference.data());

    g_device_ptr_t dptr = 0;
    KMM_GPU_CHECK(g_mem_alloc(&dptr, size));
    KMM_GPU_CHECK(g_memset_d8_async(dptr, 0, size, stream));

    memops::fill_gpu(stream, reinterpret_cast<void*>(dptr), desc);
    KMM_GPU_CHECK(g_stream_synchronize(stream));

    std::vector<std::byte> actual(size);
    KMM_GPU_CHECK(g_memcpy_d_to_h(actual.data(), dptr, size));

    KMM_GPU_CHECK(g_mem_free(dptr));
    KMM_GPU_CHECK(g_ctx_pop_current(&context));
    KMM_GPU_CHECK(g_device_primary_ctx_release(device));

    return actual == reference;
}

TEST_CASE("memops::fill_gpu", "[GPU]") {
    FillValue value = FillValue::from<int>(0x12345678);

    SECTION("scalar, no dimensions") {
        FillDescription desc(value);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("scalar at an offset") {
        FillDescription desc(value);
        desc.offset = 40;
        CHECK(check_fill_gpu(desc));
    }

    SECTION("1D contiguous") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/16, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("1D mis-aligned") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/16, /*stride=*/5);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("2D strided with offset") {
        FillDescription desc(value);
        desc.offset = 8;
        desc.add_dimension(/*extent=*/6, /*stride=*/64);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("3D") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/3, /*stride=*/400);
        desc.add_dimension(/*extent=*/4, /*stride=*/40);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("4D") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/2, /*stride=*/1500);
        desc.add_dimension(/*extent=*/3, /*stride=*/400);
        desc.add_dimension(/*extent=*/4, /*stride=*/40);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("axes contiguous (simplify merges them)") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/8, /*stride=*/16);
        desc.add_dimension(/*extent=*/4, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("axis with extent one") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/1, /*stride=*/999);
        desc.add_dimension(/*extent=*/10, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("negative stride") {
        FillDescription desc(value);
        desc.offset = 4 * 7;
        desc.add_dimension(/*extent=*/8, /*stride=*/-4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("zero stride") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/3, /*stride=*/400);
        desc.add_dimension(/*extent=*/4, /*stride=*/0);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("mixed-sign strides") {
        FillDescription desc(value);
        desc.offset = 3 * 40;
        desc.add_dimension(/*extent=*/4, /*stride=*/-40);
        desc.add_dimension(/*extent=*/9, /*stride=*/4);
        CHECK(check_fill_gpu(desc));
    }

    SECTION("wide fill value") {
        FillDescription desc(FillValue::from<double>(3.14159));
        desc.add_dimension(/*extent=*/7, /*stride=*/8);
        desc.add_dimension(/*extent=*/2, /*stride=*/64);
        CHECK(check_fill_gpu(desc));
    }
}
