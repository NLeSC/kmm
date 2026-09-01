#include <cstdint>
#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/reduction.hpp"

using namespace kmm;
using namespace kmm::memops;

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

TEST_CASE("reduce bitwise (CPU)") {
    // Reduce a (3, 2) row-major input down to 3 outputs, folding over the trailing axis.
    std::vector<int32_t> src = {
        0b1100,
        0b1010,  //
        0b0110,
        0b0011,  //
        0b1111,
        0b0101  //
    };
    std::vector<int32_t> dst(3, 0);

    auto description = [&](ReductionOp op) {
        ReductionDescription d(DataType::Int32, op);
        d.add_dimension(3, 2 * sizeof(int32_t), sizeof(int32_t));
        d.reduction_extent = 2;
        d.reduction_stride = sizeof(int32_t);
        return d;
    };

    SECTION("BitwiseAnd") {
        reduce(src.data(), dst.data(), description(ReductionOp::BitwiseAnd));
        CHECK(dst == std::vector<int32_t> {0b1000, 0b0010, 0b0101});
    }

    SECTION("BitwiseOr") {
        reduce(src.data(), dst.data(), description(ReductionOp::BitwiseOr));
        CHECK(dst == std::vector<int32_t> {0b1110, 0b0111, 0b1111});
    }

    SECTION("rejects a floating-point data type") {
        auto d = description(ReductionOp::BitwiseAnd);
        d.dtype = DataType::Float32;
        CHECK_THROWS(reduce(src.data(), dst.data(), d));
    }
}

TEST_CASE("reduce key-value / argmax (CPU)") {
    // Fold 4 (key, value) pairs down to a single pair.
    std::vector<KeyValue<double>> src = {
        {0, 3.0},
        {1, 7.0},
        {2, 1.0},
        {3, 5.0},
    };
    std::vector<KeyValue<double>> dst(1);

    ReductionDescription description(DataType::KeyValueFloat64, ReductionOp::Max);
    description.reduction_extent = 4;
    description.reduction_stride = sizeof(KeyValue<double>);

    SECTION("Max keeps the largest value with its key") {
        reduce(src.data(), dst.data(), description);
        CHECK(dst[0] == KeyValue<double> {1, 7.0});
    }

    SECTION("Min keeps the smallest value with its key") {
        description.operation = ReductionOp::Min;
        reduce(src.data(), dst.data(), description);
        CHECK(dst[0] == KeyValue<double> {2, 1.0});
    }

    SECTION("rejects an unsupported operator") {
        description.operation = ReductionOp::Sum;
        CHECK_THROWS(reduce(src.data(), dst.data(), description));
    }
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
