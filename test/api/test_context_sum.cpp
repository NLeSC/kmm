#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/reduction.hpp"

using namespace kmm;
using namespace kmm::memops;

namespace {
// Minimal duck-typed stand-in for `kmm::Layout`, avoiding a dependency on `kmm/core/layout.hpp`
// (mirrors `FakeLayout` in test/memops/test_copy.cpp).
template<size_t N>
struct FakeLayout {
    static constexpr size_t rank = N;

    ptrdiff_t offset = 0;
    ptrdiff_t extents[N];
    ptrdiff_t strides[N];
    ptrdiff_t origins[N] = {};

    ptrdiff_t base_offset() const {
        return offset;
    }

    ptrdiff_t extent(size_t axis) const {
        return extents[axis];
    }

    ptrdiff_t stride(size_t axis) const {
        return strides[axis];
    }

    ptrdiff_t begin(size_t axis) const {
        return origins[axis];
    }
};
}  // namespace

TEST_CASE("make_reduction_description") {
    // A contiguous row-major (4, 3) source reduced over its trailing axis into a contiguous
    // 4-element destination.
    FakeLayout<1> dst {/* offset */ 0, /* extents */ {4}, /* strides */ {1}};
    FakeLayout<2> src {/* offset */ 0, /* extents */ {4, 3}, /* strides */ {3, 1}};

    ReductionDescription description =
        make_reduction_description(dst, src, 1, DataType::Float32, ReductionOp::Sum);

    CHECK(description.dtype == DataType::Float32);
    CHECK(description.operation == ReductionOp::Sum);
    CHECK(description.input_offset == 0);
    CHECK(description.output_offset == 0);
    CHECK(description.reduction_extent == 3);
    CHECK(description.reduction_stride == static_cast<ptrdiff_t>(sizeof(float)));

    REQUIRE(description.num_dims == 1);
    CHECK(description.dims[0].extent == 4);
    CHECK(description.dims[0].input_stride == 3 * static_cast<ptrdiff_t>(sizeof(float)));
    CHECK(description.dims[0].output_stride == static_cast<ptrdiff_t>(sizeof(float)));
}

TEST_CASE("make_reduction_description (leading axis)") {
    // The same (4, 3) source, but now reduced over its leading axis into a contiguous
    // 3-element destination.
    FakeLayout<1> dst {/* offset */ 0, /* extents */ {3}, /* strides */ {1}};
    FakeLayout<2> src {/* offset */ 0, /* extents */ {4, 3}, /* strides */ {3, 1}};

    ReductionDescription description =
        make_reduction_description(dst, src, 0, DataType::Float32, ReductionOp::Sum);

    CHECK(description.reduction_extent == 4);
    CHECK(description.reduction_stride == 3 * static_cast<ptrdiff_t>(sizeof(float)));

    REQUIRE(description.num_dims == 1);
    CHECK(description.dims[0].extent == 3);
    CHECK(description.dims[0].input_stride == static_cast<ptrdiff_t>(sizeof(float)));
    CHECK(description.dims[0].output_stride == static_cast<ptrdiff_t>(sizeof(float)));
}

TEST_CASE("make_reduction_description end-to-end (CPU)") {
    // Reduce a (4, 3) row-major input down to 4 outputs by summing over the trailing axis, going
    // through `make_reduction_description` instead of building the description by hand (compare
    // to the "reduce (CPU)" case in test/memops/test_reduction.cpp).
    std::vector<float> src = {
        1,
        2,
        3,  //
        4,
        5,
        6,  //
        7,
        8,
        9,  //
        10,
        11,
        12  //
    };
    std::vector<float> dst(4, 0.0f);

    FakeLayout<1> dst_layout {/* offset */ 0, /* extents */ {4}, /* strides */ {1}};
    FakeLayout<2> src_layout {/* offset */ 0, /* extents */ {4, 3}, /* strides */ {3, 1}};

    ReductionDescription description = make_reduction_description(  //
        dst_layout,
        src_layout,
        1,
        DataType::Float32,
        ReductionOp::Sum
    );

    reduce(src.data(), dst.data(), description);

    CHECK(dst == std::vector<float> {6, 15, 24, 33});
}
