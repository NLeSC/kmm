#include <stdexcept>
#include <type_traits>

#include "catch2/catch_all.hpp"

#include "kmm/core/integer_fun.hpp"

using namespace kmm;

using i32 = signed int;
using u32 = unsigned int;

TEST_CASE("div_floor") {
    CHECK(div_floor(i32(0), i32(3)) == 0);
    CHECK(div_floor(i32(10), i32(3)) == 3);
    CHECK(div_floor(i32(10), i32(1)) == 10);
    CHECK(div_floor(i32(10), i32(103)) == 0);

    CHECK(div_floor(i32(0), i32(-3)) == 0);
    CHECK(div_floor(i32(10), i32(-3)) == -4);
    CHECK(div_floor(i32(10), i32(-1)) == -10);
    CHECK(div_floor(i32(10), i32(-103)) == -1);

    CHECK(div_floor(i32(-10), i32(3)) == -4);
    CHECK(div_floor(i32(-10), i32(1)) == -10);
    CHECK(div_floor(i32(-10), i32(103)) == -1);
    CHECK(div_floor(i32(-10), i32(-3)) == 3);
    CHECK(div_floor(i32(-10), i32(-1)) == 10);
    CHECK(div_floor(i32(-10), i32(-103)) == 0);

    CHECK(div_floor(u32(0), u32(3)) == 0);
    CHECK(div_floor(u32(10), u32(3)) == 3);
    CHECK(div_floor(u32(10), u32(103)) == 0);

    CHECK(div_floor(INT_MIN, INT_MIN) == 1);
    CHECK(div_floor(INT_MAX, INT_MIN) == -1);
    CHECK(div_floor(INT_MIN, INT_MAX) == -2);
    CHECK(div_floor(INT_MAX, INT_MAX) == 1);

    CHECK(div_floor(INT_MIN, 5) == INT_MIN / 5 - 1);
    CHECK(div_floor(INT_MAX, 5) == INT_MAX / 5);
    CHECK(div_floor(5, INT_MIN) == -1);
    CHECK(div_floor(5, INT_MAX) == 0);

    CHECK(div_floor(u32(0), UINT_MAX) == 0);
    CHECK(div_floor(UINT_MAX, UINT_MAX) == 1);
    CHECK(div_floor(UINT_MAX, u32(5)) == UINT_MAX / 5);
    CHECK(div_floor(u32(5), UINT_MAX) == 0);
}

TEST_CASE("div_ceil") {
    CHECK(div_ceil(i32(0), i32(3)) == 0);
    CHECK(div_ceil(i32(10), i32(3)) == 4);
    CHECK(div_ceil(i32(10), i32(1)) == 10);
    CHECK(div_ceil(i32(10), i32(103)) == 1);

    CHECK(div_ceil(i32(0), i32(-3)) == 0);
    CHECK(div_ceil(i32(10), i32(-3)) == -3);
    CHECK(div_ceil(i32(10), i32(-1)) == -10);
    CHECK(div_ceil(i32(10), i32(-103)) == 0);

    CHECK(div_ceil(i32(-10), i32(3)) == -3);
    CHECK(div_ceil(i32(-10), i32(1)) == -10);
    CHECK(div_ceil(i32(-10), i32(103)) == 0);
    CHECK(div_ceil(i32(-10), i32(-3)) == 4);
    CHECK(div_ceil(i32(-10), i32(-1)) == 10);
    CHECK(div_ceil(i32(-10), i32(-103)) == 1);

    CHECK(div_ceil(u32(0), u32(3)) == 0);
    CHECK(div_ceil(u32(10), u32(3)) == 4);
    CHECK(div_ceil(u32(10), u32(103)) == 1);

    CHECK(div_ceil(INT_MIN, INT_MIN) == 1);
    CHECK(div_ceil(INT_MAX, INT_MIN) == 0);
    CHECK(div_ceil(INT_MIN, INT_MAX) == -1);
    CHECK(div_ceil(INT_MAX, INT_MAX) == 1);

    CHECK(div_ceil(INT_MIN, 5) == INT_MIN / 5);
    CHECK(div_ceil(INT_MAX, 5) == INT_MAX / 5 + 1);
    CHECK(div_ceil(5, INT_MIN) == 0);
    CHECK(div_ceil(5, INT_MAX) == 1);

    CHECK(div_ceil(u32(0), UINT_MAX) == 0);
    CHECK(div_ceil(UINT_MAX, UINT_MAX) == 1);
    CHECK(div_ceil(UINT_MAX, u32(5)) == UINT_MAX / 5);
    CHECK(div_ceil(u32(5), UINT_MAX) == 1);
}

TEST_CASE("round_up_to_multiple") {
    CHECK(round_up_to_multiple(i32(5), i32(3)) == 6);
    CHECK(round_up_to_multiple(i32(5), i32(-3)) == 6);
    CHECK(round_up_to_multiple(i32(-5), i32(3)) == -3);
    CHECK(round_up_to_multiple(i32(-5), i32(-3)) == -3);

    CHECK(round_up_to_multiple(i32(0), i32(3)) == 0);
    CHECK(round_up_to_multiple(i32(9), i32(3)) == 9);
    CHECK(round_up_to_multiple(i32(10), i32(3)) == 12);
    CHECK(round_up_to_multiple(i32(10), i32(1)) == 10);
    CHECK(round_up_to_multiple(i32(10), i32(103)) == 103);

    CHECK(round_up_to_multiple(i32(-9), i32(3)) == -9);
    CHECK(round_up_to_multiple(i32(-10), i32(3)) == -9);
    CHECK(round_up_to_multiple(i32(-1), i32(3)) == 0);
    CHECK(round_up_to_multiple(i32(-12), i32(4)) == -12);
    CHECK(round_up_to_multiple(i32(-13), i32(4)) == -12);

    CHECK(round_up_to_multiple(i32(0), i32(-3)) == 0);
    CHECK(round_up_to_multiple(i32(9), i32(-3)) == 9);
    CHECK(round_up_to_multiple(i32(10), i32(-3)) == 12);
    CHECK(round_up_to_multiple(i32(10), i32(-1)) == 10);
    CHECK(round_up_to_multiple(i32(10), i32(-103)) == 103);

    CHECK(round_up_to_multiple(i32(-9), i32(-3)) == -9);
    CHECK(round_up_to_multiple(i32(-10), i32(-3)) == -9);
    CHECK(round_up_to_multiple(i32(-1), i32(-3)) == 0);
    CHECK(round_up_to_multiple(i32(-12), i32(-4)) == -12);
    CHECK(round_up_to_multiple(i32(-13), i32(-4)) == -12);

    CHECK(round_up_to_multiple(u32(0), u32(3)) == 0);
    CHECK(round_up_to_multiple(u32(9), u32(3)) == 9);
    CHECK(round_up_to_multiple(u32(10), u32(3)) == 12);
    CHECK(round_up_to_multiple(u32(10), u32(1)) == 10);
    CHECK(round_up_to_multiple(u32(10), u32(103)) == 103);

    CHECK(round_up_to_multiple(INT_MIN, INT_MAX) == INT_MIN + 1);
    CHECK(round_up_to_multiple(INT_MIN + 1, INT_MAX) == INT_MIN + 1);
    CHECK(round_up_to_multiple(-1, INT_MAX) == 0);
    CHECK(round_up_to_multiple(0, INT_MAX) == 0);
    CHECK(round_up_to_multiple(1, INT_MAX) == INT_MAX);
    CHECK(round_up_to_multiple(INT_MAX - 1, INT_MAX) == INT_MAX);
    CHECK(round_up_to_multiple(INT_MAX, INT_MAX) == INT_MAX);

    CHECK(round_up_to_multiple(INT_MIN, INT_MIN) == INT_MIN);
    CHECK(round_up_to_multiple(INT_MIN + 1, INT_MIN) == 0);
    CHECK(round_up_to_multiple(-1, INT_MAX) == 0);
    CHECK(round_up_to_multiple(0, INT_MIN) == 0);

    CHECK(round_up_to_multiple(u32(0), UINT_MAX) == 0);
    CHECK(round_up_to_multiple(u32(1), UINT_MAX) == UINT_MAX);
    CHECK(round_up_to_multiple(UINT_MAX - u32(1), UINT_MAX) == UINT_MAX);
    CHECK(round_up_to_multiple(UINT_MAX, UINT_MAX) == UINT_MAX);

    //  -1 * INT_MIN >= INT_MAX, so there will overflow
    CHECK_THROWS(round_up_to_multiple(1, INT_MIN));
    CHECK_THROWS(round_up_to_multiple(INT_MAX - 1, INT_MIN));
    CHECK_THROWS(round_up_to_multiple(INT_MAX, INT_MIN));

    // INT_MAX/UINT_MAX are odd, so rounding up to a multiple of 2 overflows.
    CHECK_THROWS(round_up_to_multiple(INT_MAX, 2));
    CHECK_THROWS(round_up_to_multiple(UINT_MAX, u32(2)));
}

TEST_CASE("unsigned_abs") {
    CHECK(unsigned_abs(i32(0)) == u32(0));
    CHECK(unsigned_abs(i32(5)) == u32(5));
    CHECK(unsigned_abs(i32(-5)) == u32(5));
    CHECK(unsigned_abs(INT_MAX) == u32(INT_MAX));
    CHECK(unsigned_abs(INT_MIN) == u32(INT_MAX) + u32(1));

    CHECK(unsigned_abs(u32(0)) == u32(0));
    CHECK(unsigned_abs(u32(5)) == u32(5));
    CHECK(unsigned_abs(UINT_MAX) == UINT_MAX);

    STATIC_REQUIRE(std::is_same_v<decltype(unsigned_abs(i32(0))), u32>);
}

TEST_CASE("is_divisible") {
    CHECK(is_divisible(10, 5));
    CHECK(is_divisible(10, 2));
    CHECK_FALSE(is_divisible(10, 3));

    CHECK(is_divisible(0, 5));
    CHECK(is_divisible(-10, 5));
    CHECK(is_divisible(10, -5));
    CHECK_FALSE(is_divisible(-10, 3));

    // division by zero yields false rather than throwing
    CHECK_FALSE(is_divisible(10, 0));
    CHECK_FALSE(is_divisible(0, 0));

    // mixed signed/unsigned operands
    CHECK(is_divisible(10, 5u));
    CHECK(is_divisible(-10, 5u));
    CHECK_FALSE(is_divisible(-10, 3u));

    // large unsigned operands whose remainder does not fit in int64_t
    CHECK(is_divisible(ULONG_MAX, ULONG_MAX));
    CHECK_FALSE(is_divisible(ULONG_MAX, ULONG_MAX - 1));
}

TEST_CASE("round_up_to_power_of_two") {
    // <1 always becomes 1
    CHECK(round_up_to_power_of_two(INT_MIN) == 1);
    CHECK(round_up_to_power_of_two(-1) == 1);
    CHECK(round_up_to_power_of_two(0) == 1);

    CHECK(round_up_to_power_of_two(5) == 8);
    CHECK(round_up_to_power_of_two(100) == 128);
    CHECK(round_up_to_power_of_two(128) == 128);
    CHECK(round_up_to_power_of_two(1000) == 1024);

    CHECK(round_up_to_power_of_two(u32(0)) == 1);
    CHECK(round_up_to_power_of_two(u32(5)) == 8);
    CHECK(round_up_to_power_of_two(u32(100)) == 128);
    CHECK(round_up_to_power_of_two(u32(128)) == 128);
    CHECK(round_up_to_power_of_two(u32(1000)) == 1024);
    CHECK(round_up_to_power_of_two(u32(INT_MAX)) == u32(INT_MAX) + u32(1));
    CHECK(round_up_to_power_of_two(u32(INT_MAX) + u32(1)) == u32(INT_MAX) + u32(1));

    // These should overflow
    CHECK_THROWS(round_up_to_power_of_two(INT_MAX));
    CHECK_THROWS(round_up_to_power_of_two(u32(INT_MAX) + u32(2)));
    CHECK_THROWS(round_up_to_power_of_two(UINT_MAX - 1));
    CHECK_THROWS(round_up_to_power_of_two(UINT_MAX));
}