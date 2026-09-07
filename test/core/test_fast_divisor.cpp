#include <stdexcept>

#include "catch2/catch_all.hpp"

#include "kmm/core/fast_divisor.hpp"

using namespace kmm;

using u32 = uint32_t;
using u64 = uint64_t;

TEST_CASE("FastDivisor<uint32_t>") {
    for (u32 divisor : {u32(1), u32(2), u32(3), u32(7), u32(64), u32(1000), u32(1u << 30)}) {
        FastDivisor<u32> fd(divisor);
        CAPTURE(divisor);

        CHECK(fd.get() == divisor);

        for (u32 n : {u32(0), u32(1), divisor - 1, divisor + 1, u32(12345), u32(0x7fffffff)}) {
            CAPTURE(n);
            CHECK(fd.divide(n) == n / divisor);
            CHECK(fd.modulo(n) == n % divisor);
            CHECK((n / fd) == n / divisor);
            CHECK((n % fd) == n % divisor);
        }
    }

    CHECK(FastDivisor<u32>().divide(777) == 777);
    CHECK_THROWS_AS(FastDivisor<u32>(0), std::runtime_error);
}

TEST_CASE("FastDivisor<uint64_t>") {
    for (u64 divisor : {u64(1), u64(2), u64(3), u64(7), u64(64), u64(1000), u64(1u << 30)}) {
        FastDivisor<u64> fd(divisor);
        CAPTURE(divisor);

        CHECK(fd.get() == divisor);

        for (u64 n : {u64(0), u64(1), divisor - 1, divisor + 1, u64(12345), u64(0x7fffffff)}) {
            CAPTURE(n);
            CHECK(fd.divide(n) == n / divisor);
            CHECK(fd.modulo(n) == n % divisor);
            CHECK((n / fd) == n / divisor);
            CHECK((n % fd) == n % divisor);
        }
    }

    CHECK(FastDivisor<u64>().divide(777) == 777);
    CHECK_THROWS_AS(FastDivisor<u64>(0), std::runtime_error);
}

static std::vector<uint32_t> test_extents = {
    1,
    2,
    3,
    4,
    100,
    1337,
    INT_MAX / 3,
    INT_MAX / 2 - 1,
    INT_MAX / 2,
    INT_MAX / 2 + 1,
    INT_MAX - 2,
    INT_MAX - 1,
    0xc8000000,
    INT_MAX
};

static std::vector<uint32_t> test_indices = {
    0,
    1,
    2,
    3,
    4,
    100,
    200,
    1337,
    100000,
    INT_MAX / 3,
    INT_MAX / 2 - 1,
    INT_MAX / 2,
    INT_MAX / 2 + 1,
    INT_MAX - 2,
    INT_MAX - 1,
    0xc8000000,
    INT_MAX
};

TEST_CASE("IndexMapper<uint32_t, 0>") {
    IndexMapper<uint32_t, 0> mapper {};
    CHECK(mapper.ravel({}) == 0);
    Vec<uint32_t, 0> p;

    for (uint32_t index : test_indices) {
        CAPTURE(index);
        CHECK(mapper.unravel(index, p) == (index == 0));
    }
}

TEST_CASE("IndexMapper<uint32_t, 1>") {
    for (uint32_t extent : test_extents) {
        // Total volume cannot exceed max_numerator.
        if (extent >= FastDivisor<uint32_t>::max_numerator) {
            continue;
        }

        CAPTURE(extent);

        IndexMapper<uint32_t, 1> mapper {{extent}};
        Vec<uint32_t, 1> p {};

        // test unravel -> ravel
        for (uint32_t linear_index : test_indices) {
            if (linear_index < extent) {
                CHECK(mapper.unravel(linear_index, p));
                CHECK(p[0] == linear_index);
                CHECK(mapper.ravel(p) == linear_index);
            } else {
                CHECK_FALSE(mapper.unravel(linear_index, p));
            }
        }

        // test ravel -> unravel
        for (uint32_t index : test_indices) {
            CAPTURE(index);

            if (index < extent) {
                CHECK(mapper.ravel({index}) == index);
                CHECK(mapper.unravel(index, p));
                CHECK(p[0] == index);
            }
        }
    }
}

TEST_CASE("IndexMapper<uint32_t, 2>") {
    for (uint32_t extent0 : test_extents) {
        for (uint32_t extent1 : test_extents) {
            // Total volume cannot exceed uint32_t max.
            unsigned __int128 volume = (unsigned __int128)(extent0)*extent1;

            if (volume > std::numeric_limits<uint32_t>::max()) {
                continue;
            }

            CAPTURE(extent0);
            CAPTURE(extent1);

            IndexMapper<uint32_t, 2> mapper {{extent0, extent1}};
            Vec<uint32_t, 2> p {0, 0};

            // test unravel -> ravel
            for (uint32_t linear_index : test_indices) {
                CAPTURE(linear_index);

                if (linear_index < volume) {
                    CHECK(mapper.unravel(linear_index, p));
                    CHECK(p[0] == linear_index % extent0);
                    CHECK(p[1] == linear_index / extent0);
                    CHECK(mapper.ravel(p) == linear_index);
                } else {
                    CHECK_FALSE(mapper.unravel(linear_index, p));
                }
            }

            // test ravel -> unravel
            for (uint32_t index0 : test_indices) {
                for (uint32_t index1 : test_indices) {
                    if (index0 < extent0 && index1 < extent1) {
                        uint32_t linear_index = index0 + index1 * extent0;
                        CAPTURE(index0);
                        CAPTURE(index1);

                        CHECK(mapper.ravel(Vec {index0, index1}) == linear_index);
                        CHECK(mapper.unravel(uint32_t(linear_index), p));
                        CHECK(p[0] == index0);
                        CHECK(p[1] == index1);
                    }
                }
            }
        }
    }
}
