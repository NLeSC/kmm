#include "catch2/catch_all.hpp"

#include "kmm/core/distribution.hpp"

using namespace kmm;

TEST_CASE("Distribution grid/extent/offset") {
    Distribution<2> dist(Shape<2>(100, 100), Shape<2>(32, 32));

    SECTION("evenly-divisible chunk shape yields ceil(total / chunk) grid") {
        CHECK(dist.total_shape() == Shape<2>(100, 100));
        CHECK(dist.chunk_shape() == Shape<2>(32, 32));
        CHECK(dist.grid_shape() == Shape<2>(4, 4));
        CHECK(dist.num_chunks() == 16);
    }

    SECTION("interior chunks have the nominal extent") {
        CHECK(dist.chunk_extent(Point<2>(0, 0)) == Shape<2>(32, 32));
        CHECK(dist.chunk_extent(Point<2>(1, 2)) == Shape<2>(32, 32));
    }

    SECTION("edge chunks are clipped to what remains") {
        CHECK(dist.chunk_extent(Point<2>(3, 0)) == Shape<2>(4, 32));
        CHECK(dist.chunk_extent(Point<2>(3, 3)) == Shape<2>(4, 4));
    }

    SECTION("chunk_offset is the grid index scaled by the nominal chunk shape") {
        CHECK(dist.chunk_offset(Point<2>(0, 0)) == Point<2>(0, 0));
        CHECK(dist.chunk_offset(Point<2>(3, 1)) == Point<2>(96, 32));
    }
}

TEST_CASE("Distribution::linear_index and Distribution::unravel are inverses") {
    Distribution<3> dist(Shape<3>(10, 20, 30), Shape<3>(3, 7, 11));

    for (size_t linear = 0; linear < dist.num_chunks(); linear++) {
        auto grid_index = dist.unravel(linear);
        CHECK(dist.linear_index(grid_index) == linear);
    }
}

TEST_CASE("Distribution handles a single chunk covering the whole domain") {
    Distribution<2> dist(Shape<2>(50, 50), Shape<2>(100, 100));

    CHECK(dist.grid_shape() == Shape<2>(1, 1));
    CHECK(dist.num_chunks() == 1);
    CHECK(dist.chunk_extent(Point<2>(0, 0)) == Shape<2>(50, 50));
    CHECK(dist.chunk_offset(Point<2>(0, 0)) == Point<2>(0, 0));
}
