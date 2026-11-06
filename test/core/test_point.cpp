#include <stdexcept>
#include <type_traits>

#include "catch2/catch_all.hpp"

#include "kmm/core/point.hpp"

using namespace kmm;

TEST_CASE("Point construction") {
    SECTION("default constructor") {
        Point<3> p;
        CHECK(p[0] == 0);
        CHECK(p[1] == 0);
        CHECK(p[2] == 0);
    }

    SECTION("variadic constructor") {
        Point<3> p(1, 2, 3);
        CHECK(p[0] == 1);
        CHECK(p[1] == 2);
        CHECK(p[2] == 3);
    }

    SECTION("Vec constructor") {
        Vec<default_index_type, 2> v {4, 5};
        Point<2> p(v);
        CHECK(p[0] == 4);
        CHECK(p[1] == 5);
    }

    SECTION("deduction guide") {
        auto p = Point(1, 2, 3, 4);
        static_assert(std::is_same_v<decltype(p), Point<4, default_index_type>>);
        CHECK(p[3] == 4);
    }

    SECTION("point() function") {
        auto p = point(1, 2, 3);
        CHECK(p == Point<3, int>(1, 2, 3));
    }
}

TEST_CASE("Point::one/zero") {
    CHECK(Point<3>::one() == Point<3>(1, 1, 1));
    CHECK(Point<3>::zero() == Point<3>(0, 0, 0));
}

TEST_CASE("Point::from") {
    SECTION("same dimensionality copies values") {
        Vec<int, 3> v {1, 2, 3};
        CHECK(Point<3, int>::from(v) == Point<3>(1, 2, 3));
    }

    SECTION("growing dimensionality pads with zero") {
        Vec<int, 2> v {1, 2};
        CHECK(Point<4, int>::from(v) == Point<4>(1, 2, 0, 0));
    }

    SECTION("shrinking dimensionality truncates") {
        Vec<int, 4> v {1, 2, 3, 4};
        CHECK(Point<2, int>::from(v) == Point<2>(1, 2));
    }
}

TEST_CASE("Point conversion") {
    SECTION("same dimensionality, widening type") {
        Point<2, int> src(1, 2);
        Point<2, long> dst(src);
        CHECK(dst == Point<2, long>(1, 2));
    }

    SECTION("smaller dimensionality") {
        Point<3, int> src0(1, 2, 0);
        Point<2, int> dst(src0);
        CHECK(dst == Point<2, int>(1, 2));

        Point<3, int> src1(1, 2, 3);
        CHECK_THROWS_AS((Point<2, int>(src1)), std::overflow_error);
    }

    SECTION("narrow type") {
        Point<2, int> src0(300, 1);
        CHECK_THROWS_AS((Point<2, signed char>(src0)), std::overflow_error);

        Point<2, int> src1(1, 2);
        Point<2, signed char> dst(src1);
        CHECK(dst[0] == 1);
        CHECK(dst[1] == 2);
    }
}

TEST_CASE("Point::is_convertible_to") {
    CHECK(Point<2, int>(1, 2).is_convertible_to<2, long>());
    CHECK_FALSE(Point<2, int>(300, 1).is_convertible_to<2, signed char>());
    CHECK(Point<3, int>(1, 2, 0).is_convertible_to<2, int>());
    CHECK_FALSE(Point<3, int>(1, 2, 3).is_convertible_to<2, int>());
}

TEST_CASE("Point::get_or_default") {
    Point<2> p(3, 4);
    CHECK(p.get_or_default(0) == 3);
    CHECK(p.get_or_default(1) == 4);
    CHECK(p.get_or_default(2) == 0);
    CHECK(p.get_or_default(2, 99) == 99);

    Point<0> empty;
    CHECK(empty.get_or_default(0) == 0);
    CHECK(empty.get_or_default(0, 42) == 42);
}

TEST_CASE("operator==(Point, Point") {
    CHECK(Point<2>(1, 2) == Point<2>(1, 2));
    CHECK(Point<2>(1, 2) != Point<2>(1, 3));

    // different N
    CHECK(Point<2>(1, 2) == Point<3>(1, 2, 0));
    CHECK(Point<2>(1, 2) != Point<3>(1, 2, 3));

    // different N and T
    CHECK(Point<2>(1, 2) == Point<3, short>(short(1), short(2), short(0)));
    CHECK(Point<2>(1, 2) != Point<3, short>(short(1), short(2), short(3)));
}

TEST_CASE("concat(Point, Point)") {
    Point<2, int> a(1, 2);
    Point<3, int> b(3, 4, 5);

    Point<5, int> c = concat(a, b);
    CHECK(c == Point<5, int>(1, 2, 3, 4, 5));
}
