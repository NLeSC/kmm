#include <climits>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/core/checked_math.hpp"

using namespace kmm;

TEST_CASE("checked_add") {
    CHECK(checked_add(2U, 3U) == 5);
    CHECK(checked_add(-2, 3) == 1);
    CHECK(checked_add(2UL, 3UL) == 5);
    CHECK(checked_add(-2L, 3L) == 1);

    CHECK(checked_add(INT_MAX - 1, 1) == INT_MAX);
    CHECK(checked_add(INT_MIN + 1, -1) == INT_MIN);
    CHECK(checked_add(UINT_MAX - 1, 1U) == UINT_MAX);

    CHECK_THROWS(checked_add(INT_MAX, 1));
    CHECK_THROWS(checked_add(INT_MIN, -1));
    CHECK_THROWS(checked_add(UINT_MAX, 1U));

    CHECK_THROWS(checked_add(INT_MAX, INT_MAX));
    CHECK_THROWS(checked_add(INT_MIN, INT_MIN));
    CHECK_THROWS(checked_add(UINT_MAX, UINT_MAX));

    CHECK(checked_add(LONG_MAX - 1, 1L) == LONG_MAX);
    CHECK(checked_add(LONG_MIN + 1, -1L) == LONG_MIN);
    CHECK(checked_add(ULONG_MAX - 1, 1LU) == ULONG_MAX);

    CHECK_THROWS(checked_add(LONG_MAX, 1L));
    CHECK_THROWS(checked_add(LONG_MIN, -1L));
    CHECK_THROWS(checked_add(ULONG_MAX, 1UL));

    CHECK_THROWS(checked_add(LONG_MAX, LONG_MAX));
    CHECK_THROWS(checked_add(LONG_MIN, LONG_MIN));
    CHECK_THROWS(checked_add(ULONG_MAX, ULONG_MAX));
}

TEST_CASE("checked_sub") {
    CHECK(checked_sub(5U, 3U) == 2);
    CHECK(checked_sub(3, 5) == -2);
    CHECK(checked_sub(5UL, 3UL) == 2);
    CHECK(checked_sub(3L, 5L) == -2L);

    CHECK(checked_sub(INT_MAX - 1, -1) == INT_MAX);
    CHECK(checked_sub(INT_MIN + 1, 1) == INT_MIN);
    CHECK(checked_sub(UINT_MAX, UINT_MAX) == 0U);

    CHECK_THROWS(checked_sub(INT_MAX, -1));
    CHECK_THROWS(checked_sub(INT_MIN, 1));
    CHECK_THROWS(checked_sub(0U, 1U));

    CHECK_THROWS(checked_sub(INT_MAX, INT_MIN));
    CHECK_THROWS(checked_sub(INT_MIN, INT_MAX));
    CHECK_THROWS(checked_sub(0U, UINT_MAX));

    CHECK(checked_sub(LONG_MAX - 1L, -1L) == LONG_MAX);
    CHECK(checked_sub(LONG_MIN + 1L, 1L) == LONG_MIN);
    CHECK(checked_sub(ULONG_MAX, ULONG_MAX) == 0UL);

    CHECK_THROWS(checked_sub(LONG_MAX, -1L));
    CHECK_THROWS(checked_sub(LONG_MIN, 1L));
    CHECK_THROWS(checked_sub(0UL, 1UL));

    CHECK_THROWS(checked_sub(LONG_MAX, LONG_MIN));
    CHECK_THROWS(checked_sub(LONG_MIN, LONG_MAX));
    CHECK_THROWS(checked_sub(0UL, ULONG_MAX));
}

TEST_CASE("checked_mul") {
    CHECK(checked_mul(10U, 3U) == 30);
    CHECK(checked_mul(-10, 3) == -30);
    CHECK(checked_mul(10UL, 3UL) == 30);
    CHECK(checked_mul(-10L, 3L) == -30L);
    CHECK(checked_mul(10, 0) == 0);
    CHECK(checked_mul(-10, 0) == 0);

    CHECK(checked_mul(INT_MAX, 1) == INT_MAX);
    CHECK(checked_mul(INT_MAX, -1) == INT_MIN + 1);
    CHECK(checked_mul(INT_MIN, 1) == INT_MIN);
    CHECK(checked_mul(UINT_MAX, 1U) == UINT_MAX);

    CHECK_THROWS(checked_mul(INT_MAX, 2));
    CHECK_THROWS(checked_mul(INT_MIN, 2));
    CHECK_THROWS(checked_mul(INT_MIN, -1));
    CHECK_THROWS(checked_mul(UINT_MAX, 2U));

    CHECK_THROWS(checked_mul(INT_MAX, INT_MAX));
    CHECK_THROWS(checked_mul(INT_MIN, INT_MIN));
    CHECK_THROWS(checked_mul(UINT_MAX, UINT_MAX));

    CHECK(checked_mul(LONG_MAX, 1L) == LONG_MAX);
    CHECK(checked_mul(LONG_MIN, 1L) == LONG_MIN);
    CHECK(checked_mul(ULONG_MAX, 1UL) == ULONG_MAX);

    CHECK_THROWS(checked_mul(LONG_MAX, 2L));
    CHECK_THROWS(checked_mul(LONG_MIN, 2L));
    CHECK_THROWS(checked_mul(ULONG_MAX, 2UL));

    CHECK_THROWS(checked_mul(LONG_MAX, LONG_MIN));
    CHECK_THROWS(checked_mul(LONG_MIN, LONG_MAX));
    CHECK_THROWS(checked_mul(ULONG_MAX, ULONG_MAX));
}

TEST_CASE("checked_div") {
    CHECK(checked_div(10, 3) == 3);
    CHECK(checked_div(-10, 3) == -3);

    // division by zero
    CHECK_THROWS(checked_div(10, 0));

    // overflow
    CHECK_THROWS(checked_div(INT_MIN, -1));
}

TEST_CASE("checked_rem") {
    CHECK(checked_rem(10, 3) == 1);
    CHECK(checked_rem(-10, 3) == -1);
    CHECK_THROWS(checked_rem(10, 0));
}

TEST_CASE("checked_neg") {
    CHECK(checked_neg(5) == -5);
    CHECK(checked_neg(-5) == 5);
    CHECK_THROWS(checked_neg(INT_MIN));
}

TEST_CASE("checked_abs") {
    CHECK(checked_abs(5) == 5);
    CHECK(checked_abs(-5) == 5);
    CHECK(checked_abs(0) == 0);
    CHECK_THROWS(checked_abs(INT_MIN));
}

TEST_CASE("checked_sum") {
    std::vector<int> a = {};
    CHECK(checked_sum(a.begin(), a.end()) == 0);

    a = {1, 2, 3, 4};
    CHECK(checked_sum(a.begin(), a.end()) == 10);

    a = {1, 2, 3, 4};
    CHECK(checked_sum(a.begin(), a.end(), 5) == 15);

    a = {1, 2, 3, 4, INT_MAX};
    CHECK_THROWS(checked_sum(a.begin(), a.end()));

    a = {1, 2, 3, 4, INT_MAX};
    CHECK(checked_sum(a.begin(), a.end(), long()) == long(INT_MAX) + 10);
}

TEST_CASE("checked_product") {
    std::vector<int> a = {};
    CHECK(checked_product(a.begin(), a.end()) == 1);

    a = {1, 2, 3, 4};
    CHECK(checked_product(a.begin(), a.end()) == 24);

    a = {1, 2, 3, 4};
    CHECK(checked_product(a.begin(), a.end(), 5) == 120);

    a = {1, 2, 3, 4, INT_MAX};
    CHECK_THROWS(checked_product(a.begin(), a.end()));

    a = {1, 2, 3, 4, INT_MAX};
    CHECK(checked_product(a.begin(), a.end(), long(1)) == long(INT_MAX) * 24);
}

template<typename T>
std::vector<T> generate_inputs(std::mt19937_64 rng, size_t n) {
    std::vector<T> inputs = {
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::min() + static_cast<T>(1),
        static_cast<T>(-2),
        static_cast<T>(-1),
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(2),
        std::numeric_limits<T>::max() - static_cast<T>(1),
        std::numeric_limits<T>::max(),
    };

    std::uniform_int_distribution<T> dist(
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max()
    );

    while (inputs.size() < n) {
        inputs.push_back(dist(rng));
    }

    return inputs;
}

template<typename L, typename R, typename O, typename F, typename G>
void stress_test_checked(F op, G checked_op) {
    INFO("L=" << typeid(L).name());
    INFO("R=" << typeid(R).name());
    INFO("O=" << typeid(O).name());

    auto left_inputs = generate_inputs<L>(std::mt19937_64 {0}, 1000);
    auto right_inputs = generate_inputs<R>(std::mt19937_64 {1}, 1000);

    for (auto a : left_inputs) {
        for (auto b : right_inputs) {
            O c;

            INFO("left=" << a);
            INFO("right=" << b);

            __int128 expected = op(__int128(a), __int128(b));

            if (expected >= __int128(std::numeric_limits<O>::min())
                && expected <= __int128(std::numeric_limits<O>::max())) {
                INFO("expected=" << static_cast<O>(expected));
                REQUIRE(checked_op(a, b, &c));

                INFO("gotten=" << c);
                REQUIRE(c == static_cast<O>(expected));
            } else {
                INFO("expected=<out of range>");
                REQUIRE_FALSE(checked_op(a, b, &c));
            }
        }
    }
}

template<typename L, typename R, typename O>
void stress_test_checked_add() {
    stress_test_checked<L, R, O>(std::plus<__int128> {}, detail::checked_add_impl<L, R, O>::apply);
}

TEST_CASE("checked_add (stress test)", "[.][slow]") {
    stress_test_checked_add<int, int, int>();
    stress_test_checked_add<uint, uint, uint>();
    stress_test_checked_add<long, long, long>();
    stress_test_checked_add<ulong, ulong, ulong>();

    stress_test_checked_add<long, ulong, long>();
    stress_test_checked_add<long, long, long>();
    stress_test_checked_add<long, ulong, ulong>();
    stress_test_checked_add<long, long, ulong>();
    stress_test_checked_add<ulong, ulong, long>();
    stress_test_checked_add<ulong, long, long>();
    stress_test_checked_add<ulong, ulong, ulong>();
    stress_test_checked_add<ulong, long, ulong>();
}

template<typename L, typename R, typename O>
void stress_test_checked_sub() {
    stress_test_checked<L, R, O>(std::minus<__int128> {}, detail::checked_sub_impl<L, R, O>::apply);
}

TEST_CASE("checked_sub (stress test)", "[.][slow]") {
    stress_test_checked_sub<int, int, int>();
    stress_test_checked_sub<uint, uint, uint>();
    stress_test_checked_sub<long, long, long>();
    stress_test_checked_sub<ulong, ulong, ulong>();

    stress_test_checked_sub<long, ulong, long>();
    stress_test_checked_sub<long, long, long>();
    stress_test_checked_sub<long, ulong, ulong>();
    stress_test_checked_sub<long, long, ulong>();
    stress_test_checked_sub<ulong, ulong, long>();
    stress_test_checked_sub<ulong, long, long>();
    stress_test_checked_sub<ulong, ulong, ulong>();
    stress_test_checked_sub<ulong, long, ulong>();
}

template<typename L, typename R, typename O>
void stress_test_checked_mul() {
    stress_test_checked<L, R, O>(
        [](__int128 a, __int128 b) {
            auto x = a >= 0 ? (unsigned __int128)a : (unsigned __int128)-a;
            auto y = b >= 0 ? (unsigned __int128)b : (unsigned __int128)-b;
            auto z = x * y;
            return ((a >= 0) == (b >= 0)) ? __int128(z) : -__int128(z);
        },
        detail::checked_mul_impl<L, R, O>::apply
    );
}

TEST_CASE("checked_mul (stress test)", "[.][slow]") {
    stress_test_checked_mul<int, int, int>();
    stress_test_checked_mul<uint, uint, uint>();
    stress_test_checked_mul<long, long, long>();
    stress_test_checked_mul<ulong, ulong, ulong>();

    stress_test_checked_mul<long, ulong, long>();
    stress_test_checked_mul<long, long, long>();
    stress_test_checked_mul<long, ulong, ulong>();
    stress_test_checked_mul<long, long, ulong>();
    stress_test_checked_mul<ulong, ulong, long>();
    stress_test_checked_mul<ulong, long, long>();
    stress_test_checked_mul<ulong, ulong, ulong>();
    stress_test_checked_mul<ulong, long, ulong>();
}

template<typename L, typename R, typename O>
void stress_test_checked_div() {
    stress_test_checked<L, R, O>(
        [](__int128 a, __int128 b) { return b != 0 ? a / b : __int128(1) << 127; },
        detail::checked_div_impl<L, R, O>::apply
    );
}

TEST_CASE("checked_div (stress test)", "[.][slow]") {
    stress_test_checked_div<int, int, int>();
    stress_test_checked_mul<uint, uint, uint>();
    stress_test_checked_div<long, long, long>();
    stress_test_checked_div<ulong, ulong, ulong>();

    stress_test_checked_div<long, ulong, long>();
    stress_test_checked_div<long, long, long>();
    stress_test_checked_div<long, ulong, ulong>();
    stress_test_checked_div<long, long, ulong>();
    stress_test_checked_div<ulong, ulong, long>();
    stress_test_checked_div<ulong, long, long>();
    stress_test_checked_div<ulong, ulong, ulong>();
    stress_test_checked_div<ulong, long, ulong>();
}