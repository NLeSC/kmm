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

    ReductionDescription description;
    description.dtype = DataType::Float32;
    description.operation = ReductionOp::Sum;
    description.add_dimension(4, 3 * sizeof(float), sizeof(float));
    description.reduction_extent = 3;
    description.reduction_stride = sizeof(float);

    reduce(src.data(), dst.data(), description);

    CHECK(dst == std::vector<float> {6, 15, 24, 33});
}
