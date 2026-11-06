#include "catch2/catch_all.hpp"

#include "kmm/core/vec.hpp"

using namespace kmm;

TEST_CASE("Vec<int, 0>") {
    Vec<int, 0> a;
    Vec<int, 0> b {};
    auto c = Vec<int, 0> {};

    //    CHECK(a.data() == nullptr);
    //    CHECK(b.data() == nullptr);
    //    CHECK(c.data() == nullptr);
}

TEST_CASE("Vec<T, 1>") {
    Vec<int, 1> a;
    Vec<int, 1> b {1};

    CHECK(b[0] == 1);
    CHECK(b.x == 1);
    //    CHECK(b.data() == &b.x);

    b[0] = 5;
    CHECK(b.x == 5);
}

TEST_CASE("Vec<T, 2>") {
    Vec<int, 2> a;
    Vec<int, 2> b {1, 2};

    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b.x == 1);
    CHECK(b.y == 2);

    //    CHECK(b.data() == &b.x);
    //    CHECK(b.data()[1] == b.y);

    b[1] = 9;
    CHECK(b.y == 9);
}

TEST_CASE("Vec<T, 3>") {
    Vec<int, 3> a;
    Vec<int, 3> b {1, 2, 3};

    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
    CHECK(b.x == 1);
    CHECK(b.y == 2);
    CHECK(b.z == 3);

    //    CHECK(b.data() == &b.x);
    //    CHECK(b.data()[2] == b.z);

    b[2] = 42;
    CHECK(b.z == 42);
}

TEST_CASE("Vec<T, 4>") {
    Vec<int, 4> a;
    Vec<int, 4> b {1, 2, 3, 4};

    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
    CHECK(b[3] == 4);
    CHECK(b.x == 1);
    CHECK(b.y == 2);
    CHECK(b.z == 3);
    CHECK(b.w == 4);

    //    CHECK(b.data() == &b.x);
    //    CHECK(b.data()[3] == b.w);

    b[3] = 42;
    CHECK(b.w == 42);
}

TEST_CASE("Vec<T, N> generic") {
    Vec<int, 5> a;
    Vec<int, 5> b {1, 2, 3, 4, 5};

    for (size_t i = 0; i < 5; i++) {
        CHECK(b[i] == static_cast<int>(i + 1));
    }

    //    CHECK(b.data() == b.values);

    b[4] = 100;
    CHECK(b.values[4] == 100);

    const Vec<int, 5> c {5, 4, 3, 2, 1};
    for (size_t i = 0; i < 5; i++) {
        CHECK(c[i] == static_cast<int>(5 - i));
    }
}

TEST_CASE("fill") {
    auto v0 = fill<0>(42);
    //    CHECK(v0.data() == nullptr);

    auto v1 = fill<1>(7);
    CHECK(v1[0] == 7);

    auto v2 = fill<2>(3);
    CHECK(v2[0] == 3);
    CHECK(v2[1] == 3);

    auto v4 = fill<4>(9);
    for (size_t i = 0; i < 4; i++) {
        CHECK(v4[i] == 9);
    }

    auto v5 = fill<6>(1);
    for (size_t i = 0; i < 6; i++) {
        CHECK(v5[i] == 1);
    }
}

TEST_CASE("concat") {
    auto a = Vec<int, 0> {};
    auto b = Vec {1};
    auto c = Vec {2, 3, 4};

    auto x = concat(a, b);
    CHECK(x[0] == 1);

    auto y = concat(b, c);
    CHECK(y[0] == 1);
    CHECK(y[1] == 2);
    CHECK(y[2] == 3);
    CHECK(y[3] == 4);
}
