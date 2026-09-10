#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "catch2/catch_all.hpp"

#include "kmm/core/checked_compare.hpp"

using namespace kmm;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

#define CHECK_IS_LESS(A, B)              \
    CHECK(is_less(A, B));                \
    CHECK(is_less_equal(A, B));          \
    CHECK_FALSE(is_greater_equal(A, B)); \
    CHECK_FALSE(is_greater(A, B));       \
    CHECK_FALSE(is_less(B, A));          \
    CHECK_FALSE(is_less_equal(B, A));    \
    CHECK(is_greater(B, A));             \
    CHECK(is_greater_equal(B, A));

TEST_CASE("is_less") {
    SECTION("signed vs unsigned, negative") {
        CHECK_IS_LESS(i8(-128), u8(0));
        CHECK_IS_LESS(i16(-1), u16(1));
        CHECK_IS_LESS(i32(-100), u32(0));
        CHECK_IS_LESS(i64(std::numeric_limits<i64>::min()), u64(0ULL));
        CHECK_IS_LESS(i32(-1), u32(10));
        CHECK_IS_LESS(i32(-1), i32(0));
    }

    SECTION("signed vs unsigned, positive") {
        CHECK_IS_LESS(u8(0), i8(127));
        CHECK_IS_LESS(u16(1), i16(2));
        CHECK_IS_LESS(u32(0), i64(1));
        CHECK_IS_LESS(u64(0ULL), i64(LONG_MAX));
        CHECK_IS_LESS(i32(INT_MIN), i32(0));
        CHECK_IS_LESS(i32(-2147483647), u32(2147483648u));
        CHECK_IS_LESS(i64(0), u64(ULONG_MAX));
    }

    SECTION("different integers") {
        CHECK_IS_LESS(u8(200), u16(300));
        CHECK_IS_LESS(i16(30000), i32(40000));
        CHECK_IS_LESS(i32(2147483646), i64(2147483647));
        CHECK_IS_LESS(u32(4294967294U), u64(4294967295ULL));
        CHECK_IS_LESS(i64(9223372036854775806LL), u64(9223372036854775807ULL));
    }

    SECTION("float vs float") {
        CHECK_IS_LESS(f32(1.5f), f64(1.6));
        CHECK_IS_LESS(f64(-1.0), f32(0.0f));
        CHECK_IS_LESS(f32(-std::numeric_limits<f32>::max()), f64(std::numeric_limits<f64>::max()));
        CHECK_IS_LESS(f64(-std::numeric_limits<f64>::infinity()), f64(-1e308));
        CHECK_IS_LESS(f32(-1e10f), f64(-1e9));
        CHECK_IS_LESS(f64(1.1), f64(1.2L));
        CHECK_IS_LESS(f64(-1.2L), f64(-1.1));
        CHECK_IS_LESS(f64(0.0L), f64(0.1L));
        CHECK_IS_LESS(f64(0.0), std::numeric_limits<f64>::infinity());
        CHECK_IS_LESS(f32(-std::numeric_limits<f32>::infinity()), f32(0.0f));
        CHECK_IS_LESS(f32(std::numeric_limits<f32>::max()), f64(std::numeric_limits<f64>::max()));
        CHECK_IS_LESS(f32(1.0001f), f64(1.0002));
    }

    SECTION("float vs integer, fractional values") {
        CHECK_IS_LESS(u16(0), f64(0.1));
        CHECK_IS_LESS(u64(100), f64(101.0));
        CHECK_IS_LESS(i64(-1000), f32(0.0f));
        CHECK_IS_LESS(f32(0.999f), i32(1));
        CHECK_IS_LESS(i8(-10), f32(-9.9f));
        CHECK_IS_LESS(i32(1), f64(1.1));
        CHECK_IS_LESS(i64(-1), f64(0.0));
        CHECK_IS_LESS(u32(1), f64(1.1));
        CHECK_IS_LESS(f64(-0.01), u32(0));
        CHECK_IS_LESS(f32(-0.1f), u16(0));
        CHECK_IS_LESS(u64(1), f64(1.0001L));
        CHECK_IS_LESS(i8(10), f64(10.1L));
        CHECK_IS_LESS(i16(123), f64(123.1L));
        CHECK_IS_LESS(u16(65535), f64(65535.5));
        CHECK_IS_LESS(i32(-1), f64(0.0L));

        // f64(LONG_MAX) will round up, f64(LONG_MIN + 1) will round down.
        CHECK_IS_LESS(LONG_MAX, f64(LONG_MAX));
        CHECK_IS_LESS(f64(LONG_MIN + 1), LONG_MIN + 1);
    }

    SECTION("float rules: equal values, -0.0 vs 0.0, NaN") {
        CHECK_FALSE(is_less(-0.0, 0.0));
        CHECK_FALSE(is_less(f32(1.1f), f64(1.1)));
        CHECK_FALSE(is_less(u8(255), u8(255)));
        CHECK_FALSE(is_less(std::numeric_limits<f64>::quiet_NaN(), f64(0.0)));
    }
}

#define CHECK_IS_EQUAL(A, B)       \
    CHECK(is_equal(A, B));         \
    CHECK(is_equal(B, A));         \
    CHECK_FALSE(is_less(A, B));    \
    CHECK_FALSE(is_less(B, A));    \
    CHECK(is_less_equal(A, B));    \
    CHECK(is_less_equal(B, A));    \
    CHECK(is_greater_equal(A, B)); \
    CHECK(is_greater_equal(B, A)); \
    CHECK_FALSE(is_greater(A, B)); \
    CHECK_FALSE(is_greater(B, A));

#define CHECK_NOT_EQUAL(A, B)    \
    CHECK_FALSE(is_equal(A, B)); \
    CHECK_FALSE(is_equal(B, A)); \
    CHECK((is_less(A, B) || is_less(B, A)));

TEST_CASE("is_equal") {
    SECTION("signed vs unsigned") {
        CHECK_IS_EQUAL(i8(0), u8(0));
        CHECK_IS_EQUAL(i16(123), u16(123));
        CHECK_IS_EQUAL(u32(3000), i64(3000));
        CHECK_IS_EQUAL(u64(0ULL), i64(0LL));
        CHECK_IS_EQUAL(u8(255), i32(255));
        CHECK_IS_EQUAL(u32(1), f64(1.0));
        CHECK_IS_EQUAL((unsigned long long)(42), f64(42.0L));
        CHECK_IS_EQUAL(i64(1234567890123LL), u64(1234567890123ULL));
        CHECK_IS_EQUAL(u32(0), f32(0.0f));
    }

    SECTION("different integers") {
        CHECK_IS_EQUAL(i32(-100), i64(-100));
        CHECK_IS_EQUAL(i32(INT_MAX), i64(INT_MAX));
        CHECK_IS_EQUAL(i8(-128), i16(-128));
        CHECK_IS_EQUAL(u16(65535), u32(65535));
        CHECK_IS_EQUAL(i16(-32768), i32(-32768));
    }

    SECTION("float vs float") {
        CHECK_IS_EQUAL(f32(2.5f), f64(2.5));
        CHECK_IS_EQUAL(f64(-1.0), f64(-1.0L));
        CHECK_IS_EQUAL(
            f32(std::numeric_limits<f32>::infinity()),
            f64(std::numeric_limits<f64>::infinity())
        );
        CHECK_IS_EQUAL(f64(-0.0), f32(-0.0f));
        CHECK_IS_EQUAL(f64(0.0L), f64(0.0));
        CHECK_IS_EQUAL(f32(0.5f), f64(0.5L));
        CHECK_IS_EQUAL(
            f64(-std::numeric_limits<f64>::infinity()),
            f64(-std::numeric_limits<f64>::infinity())
        );
        CHECK_IS_EQUAL(f64(123456.0L), f64(123456.0L));
    }

    SECTION("float vs integer") {
        CHECK_IS_EQUAL(i64(-1), f64(-1.0L));
        CHECK_IS_EQUAL(i32(2147483647), f64(2147483647.0));
    }

    SECTION("signed vs unsigned, not equal") {
        CHECK_NOT_EQUAL(i8(0), u8(1));
        CHECK_NOT_EQUAL(i16(123), u16(124));
        CHECK_NOT_EQUAL(i32(-100), i64(100));
        CHECK_NOT_EQUAL(u32(3000), i64(-3000));
        CHECK_NOT_EQUAL(u64(1), i64(2));
        CHECK_NOT_EQUAL(u8(255), i32(254));
        CHECK_NOT_EQUAL(u32(1), f64(2.0));
        CHECK_NOT_EQUAL((unsigned long long)(42), f64(42.1));
    }

    SECTION("integers, not equal") {
        CHECK_NOT_EQUAL(i32(INT_MAX), i64(INT_MAX) - 1);
        CHECK_NOT_EQUAL(i8(-128), i16(-127));
        CHECK_NOT_EQUAL(u16(65534), u32(65535));
        CHECK_NOT_EQUAL(i16(-32768), i32(-32767));
        CHECK_NOT_EQUAL(i64(1234567890123LL), u64(1234567890124ULL));
    }

    SECTION("floats, not equal") {
        CHECK_NOT_EQUAL(f32(2.5f), f64(2.5001));
        CHECK_NOT_EQUAL(f64(-1.0), f64(-1.0000000001L));
        CHECK_NOT_EQUAL(u64(ULONG_MAX), f64(ULONG_MAX) - 1.0);
        CHECK_NOT_EQUAL(u64(ULONG_MAX), f64(ULONG_MAX));
        CHECK_NOT_EQUAL(
            f32(std::numeric_limits<f32>::infinity()),
            f64(std::numeric_limits<f64>::lowest())
        );
        CHECK_NOT_EQUAL(f64(123.0), f64(124.0));
        CHECK_NOT_EQUAL(f64(1.0L), f64(1.0000000001L));
        CHECK_NOT_EQUAL(f64(0.0L), f64(0.1));
        CHECK_NOT_EQUAL(f32(0.5f), f64(0.5000001L));
        CHECK_NOT_EQUAL(u32(0), f32(0.0001f));
    }

    SECTION("float vs integer, not equal") {
        CHECK_NOT_EQUAL(i64(-1), f64(1.0L));
        CHECK_NOT_EQUAL(i32(2147483647), f64(2147483646.0));
    }

    SECTION("NaN is never equal") {
        CHECK_FALSE(is_equal(std::numeric_limits<f64>::quiet_NaN(), f64(0.0)));
        CHECK_FALSE(is_equal(std::numeric_limits<f64>::quiet_NaN(), i32(0)));
        CHECK_FALSE(
            is_equal(std::numeric_limits<f64>::quiet_NaN(), std::numeric_limits<f32>::quiet_NaN())
        );
    }
}

TEST_CASE("is_convertible") {
    SECTION("convertible: integer to integer") {
        CHECK(is_convertible<u32>(i32(0)));
        CHECK(is_convertible<u32>(i64(4294967295LL)));
        CHECK(is_convertible<i16>(i8(127)));
        CHECK(is_convertible<u8>(i32(255)));
        CHECK(is_convertible<i8>(i16(-128)));
        CHECK(is_convertible<u16>(u32(65535)));
        CHECK(is_convertible<i32>(i32(INT_MIN)));
        CHECK(is_convertible<i32>(u32(100)));
        CHECK(is_convertible<i64>(u64(12345ULL)));
        CHECK(is_convertible<u64>(i64(0)));
    }

    SECTION("convertible: integer to float") {
        CHECK(is_convertible<f32>(i32(123456)));
        CHECK(is_convertible<f64>(f32(1.5f)));
        CHECK(is_convertible<f32>(f64(1.5)));
        CHECK(is_convertible<f64>(INT_MIN));
        CHECK(is_convertible<f32>(std::numeric_limits<f64>::infinity()));
        CHECK(is_convertible<f64>(1.0 + std::numeric_limits<f32>::epsilon()));
    }

    SECTION("convertible: float to integer, whole value") {
        CHECK(is_convertible<i32>(f64(0.0)));
        CHECK(is_convertible<i32>(f64(2147483647.0)));
        CHECK(is_convertible<i64>(f64(9007199254740992.0)));
        CHECK(is_convertible<u64>(f64(0.0)));
        CHECK(is_convertible<i32>(f32(-100.0f)));
        CHECK(is_convertible<i64>(f64(1.0L)));
        CHECK(is_convertible<i32>(f64(-2147483648.0)));
        CHECK(is_convertible<i64>(f64(9007199254740991.0)));
        CHECK(is_convertible<i32>(f32(16777216.0f)));
        CHECK(is_convertible<u8>(f64(0.0)));
        CHECK(is_convertible<i16>(f64(-32768.0)));
    }

    SECTION("not convertible: integer overflow/underflow") {
        CHECK_FALSE(is_convertible<u32>(i32(-1)));
        CHECK_FALSE(is_convertible<i8>(i16(128)));
        CHECK_FALSE(is_convertible<u16>(i32(70000)));
        CHECK_FALSE(is_convertible<u8>(i32(256)));
        CHECK_FALSE(is_convertible<u8>(i32(-1)));
    }

    SECTION("not convertible: fractional value") {
        CHECK_FALSE(is_convertible<i32>(f64(0.1)));
        CHECK_FALSE(is_convertible<i32>(f64(2147483648.0)));
        CHECK_FALSE(is_convertible<u8>(f64(255.5)));
        CHECK_FALSE(is_convertible<i16>(f64(40000.0)));
        CHECK_FALSE(is_convertible<i64>(f64(9.223372036854775808e18)));
        CHECK_FALSE(is_convertible<u32>(f64(-0.1)));
        CHECK_FALSE(is_convertible<i32>(f32(2147483648.0f)));
        CHECK_FALSE(is_convertible<i16>(f64(-32769.0)));
        CHECK_FALSE(is_convertible<u64>(f64(-1.0)));
        CHECK_FALSE(is_convertible<i32>(f64(1.5L)));
        CHECK_FALSE(is_convertible<i64>(f64(std::numeric_limits<f64>::max())));
        CHECK_FALSE(is_convertible<i32>(f64(-2147483649.0)));
        CHECK_FALSE(is_convertible<i32>(f32(1.0000001f)));
        CHECK_FALSE(is_convertible<i32>(f64(-0.00001)));
        CHECK_FALSE(is_convertible<i32>(f64(1e20)));
        CHECK_FALSE(is_convertible<i16>(f32(32768.0f)));
        CHECK_FALSE(is_convertible<f32>(1.0 + std::numeric_limits<f64>::epsilon()));
        CHECK_FALSE(is_convertible<i32>(1.0f + std::numeric_limits<f32>::epsilon()));
        CHECK_FALSE(is_convertible<u32>(std::numeric_limits<f64>::epsilon()));
    }

    SECTION("not convertible: infinity and NaN") {
        CHECK_FALSE(is_convertible<i32>(f64(std::numeric_limits<f64>::infinity())));
        CHECK_FALSE(is_convertible<i32>(f64(std::numeric_limits<f64>::quiet_NaN())));
        CHECK_FALSE(is_convertible<u32>(f64(std::numeric_limits<f64>::infinity())));
        CHECK_FALSE(is_convertible<u8>(f64(std::numeric_limits<f64>::quiet_NaN())));
    }
}

TEST_CASE("checked_cast") {
    SECTION("widening nothrow") {
        CHECK_NOTHROW(checked_cast<u64>(INT_MAX));
        CHECK_NOTHROW(checked_cast<u64>(UINT_MAX));
        CHECK_NOTHROW(checked_cast<u64>(LONG_MAX));
        CHECK_NOTHROW(checked_cast<u64>(ULONG_MAX));

        CHECK_NOTHROW(checked_cast<i64>(INT_MAX));
        CHECK_NOTHROW(checked_cast<i64>(UINT_MAX));
        CHECK_NOTHROW(checked_cast<i64>(LONG_MAX));

        CHECK_NOTHROW(checked_cast<u32>(INT_MAX));
        CHECK_NOTHROW(checked_cast<u32>(UINT_MAX));

        CHECK_NOTHROW(checked_cast<i32>(INT_MAX));
    }

    SECTION("narrowing throw") {
        CHECK_THROWS(checked_cast<i64>(ULONG_MAX));

        CHECK_THROWS(checked_cast<u32>(LONG_MAX));
        CHECK_THROWS(checked_cast<u32>(ULONG_MAX));

        CHECK_THROWS(checked_cast<i32>(UINT_MAX));
        CHECK_THROWS(checked_cast<i32>(LONG_MAX));
        CHECK_THROWS(checked_cast<i32>(ULONG_MAX));

        CHECK_THROWS(checked_cast<i8>(i32(200)));
        CHECK_THROWS(checked_cast<i8>(f64(200.0)));
        CHECK_THROWS(checked_cast<u8>(i64(-1337)));
    }

    SECTION("float to integer nothrow") {
        CHECK_NOTHROW(checked_cast<f64>(i32(200)));
        CHECK_NOTHROW(checked_cast<u8>(f64(201.0)));
        CHECK_NOTHROW(checked_cast<u8>(i32(202)));
    }

    SECTION("integer to float throw") {
        CHECK_THROWS(checked_cast<f32>(INT_MAX));  // not representable
        CHECK_THROWS(checked_cast<i32>(5.5));  // fractional part
        CHECK_THROWS(checked_cast<u32>(-5.0));  // negative value
        CHECK_THROWS(checked_cast<i32>(std::numeric_limits<f32>::max()));
        CHECK_THROWS(checked_cast<i32>(std::numeric_limits<f32>::infinity()));
        CHECK_THROWS(checked_cast<i32>(std::numeric_limits<f32>::quiet_NaN()));
    }
}