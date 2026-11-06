#include <algorithm>
#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/fill.hpp"

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

static bool check_fill(const FillDescription& desc) {
    size_t size = size_fill(desc) * 2;  // *2 just to be sure

    std::vector<std::byte> expected(size);
    std::vector<std::byte> actual(size);
    std::vector<std::byte> simplified(size);

    test_fill(desc, expected.data());
    test_fill(desc.simplify(), simplified.data());
    memops::fill(actual.data(), desc);

    return actual == expected && simplified == expected;
}

TEST_CASE("memops::fill matches reference") {
    FillValue value = FillValue::from<int>(0x12345678);

    SECTION("scalar, no dimensions") {
        FillDescription desc(value);
        CHECK(check_fill(desc));
    }

    SECTION("scalar at an offset") {
        FillDescription desc(value);
        desc.offset = 40;
        CHECK(check_fill(desc));
    }

    SECTION("1D contiguous") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/16, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("1D mis-aligned") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/16, /*stride=*/5);
        CHECK(check_fill(desc));
    }

    SECTION("2D strided with offset") {
        FillDescription desc(value);
        desc.offset = 8;
        desc.add_dimension(/*extent=*/6, /*stride=*/64);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("3D") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/3, /*stride=*/400);
        desc.add_dimension(/*extent=*/4, /*stride=*/40);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("4D") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/2, /*stride=*/1500);
        desc.add_dimension(/*extent=*/3, /*stride=*/400);
        desc.add_dimension(/*extent=*/4, /*stride=*/40);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("axes contiguous (simplify merges them)") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/8, /*stride=*/16);
        desc.add_dimension(/*extent=*/4, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("axes contiguous (reversed order)") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/4, /*stride=*/4);
        desc.add_dimension(/*extent=*/8, /*stride=*/16);
        CHECK(check_fill(desc));
    }

    SECTION("axes overlapping (simplify merges them)") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/8, /*stride=*/16);
        desc.add_dimension(/*extent=*/4, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("axis with extent one") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/1, /*stride=*/999);
        desc.add_dimension(/*extent=*/10, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("negative stride") {
        FillDescription desc(value);
        desc.offset = 4 * 7;
        desc.add_dimension(/*extent=*/8, /*stride=*/-4);
        CHECK(check_fill(desc));
    }

    SECTION("2D negative stride") {
        FillDescription desc(value);
        desc.offset = 256;
        desc.add_dimension(/*extent=*/8, /*stride=*/-4);
        desc.add_dimension(/*extent=*/2, /*stride=*/-40);
        CHECK(check_fill(desc));
    }

    SECTION("zero stride") {
        FillDescription desc(value);
        desc.add_dimension(/*extent=*/3, /*stride=*/400);
        desc.add_dimension(/*extent=*/4, /*stride=*/0);
        desc.add_dimension(/*extent=*/5, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("mixed-sign strides") {
        FillDescription desc(value);
        desc.offset = 3 * 40;
        desc.add_dimension(/*extent=*/4, /*stride=*/-40);
        desc.add_dimension(/*extent=*/9, /*stride=*/4);
        CHECK(check_fill(desc));
    }

    SECTION("wide fill value") {
        FillDescription desc(FillValue::from<double>(3.14159));
        desc.add_dimension(/*extent=*/7, /*stride=*/8);
        desc.add_dimension(/*extent=*/2, /*stride=*/64);
        CHECK(check_fill(desc));
    }
}
