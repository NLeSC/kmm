#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/fill.hpp"

using namespace kmm;

TEST_CASE("fill (CPU)") {
    std::vector<int> dst(6, -1);

    FillDescription description(FillValue::from<int>(42));
    description.add_dimension(6, sizeof(int));

    fill(dst.data(), description);

    CHECK(dst == std::vector<int>(6, 42));
}
