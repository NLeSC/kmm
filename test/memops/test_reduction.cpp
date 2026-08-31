#include <cstdint>
#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/reduction.hpp"

using namespace kmm;

TEST_CASE("reduce (CPU)") {
    // Reduce a (4, 3) row-major input down to 4 outputs by summing over the trailing axis.
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

    ReductionDescription description(DataType::Float32, ReductionOp::Sum);
    description.add_dimension(4, 3 * sizeof(float), sizeof(float));
    description.reduction_extent = 3;
    description.reduction_stride = sizeof(float);

    reduce(src.data(), dst.data(), description);

    CHECK(dst == std::vector<float> {6, 15, 24, 33});
}

TEST_CASE("ReductionDescription::src_range/dst_range") {
    constexpr auto elem = static_cast<memops_stride_type>(sizeof(int32_t));

    SECTION("only the reduced axis") {
        ReductionDescription description;
        description.dtype = DataType::Int32;
        description.input_offset = 4;
        description.output_offset = 8;
        description.reduction_extent = 3;
        description.reduction_stride = elem;

        // src spans the 3 reduced elements; dst is a single element (the reduced axis does not
        // move the output pointer).
        CHECK(description.src_range() == Range<ptrdiff_t>(4, 4 + 3 * elem));
        CHECK(description.dst_range() == Range<ptrdiff_t>(8, 8 + elem));
    }

    SECTION("batch axis plus reduced axis") {
        ReductionDescription description;
        description.dtype = DataType::Int32;
        description.add_dimension(4, 3 * elem, elem);
        description.reduction_extent = 3;
        description.reduction_stride = elem;

        CHECK(description.src_range() == Range<ptrdiff_t>(0, 3 * 3 * elem + 2 * elem + elem));
        CHECK(description.dst_range() == Range<ptrdiff_t>(0, 3 * elem + elem));
    }

    SECTION("negative reduction stride") {
        ReductionDescription description;
        description.dtype = DataType::Int32;
        description.add_dimension(4, elem, elem);
        description.reduction_extent = 3;
        description.reduction_stride = -elem;

        // The reduced axis walks backwards, pulling the lower bound below the base offset.
        CHECK(description.src_range() == Range<ptrdiff_t>(-2 * elem, 3 * elem + elem));
        CHECK(description.dst_range() == Range<ptrdiff_t>(0, 3 * elem + elem));
    }
}
