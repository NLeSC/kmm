#include <stdexcept>

#include "catch2/catch_all.hpp"

#include "kmm/core/shape.hpp"

using namespace kmm;

template<typename DomainT>
void foo(DomainT x) {}

#include "kmm/core/bounds.hpp"

TEST_CASE("Shape construction") {
    SECTION("default constructs all zeros") {
        Shape<3, int> s;
        CHECK(s[0] == 0);
        CHECK(s[1] == 0);
        CHECK(s[2] == 0);
    }

    SECTION("variadic constructor takes exactly N extents") {
        Shape<3, int> s(4, 5, 6);
        CHECK(s[0] == 4);
        CHECK(s[1] == 5);
        CHECK(s[2] == 6);
    }

    SECTION("wraps an existing Vec unchanged") {
        Vec<int, 2> v {4, 5};
        Shape<2, int> s(v);
        CHECK(s[0] == 4);
        CHECK(s[1] == 5);
    }

    SECTION("shape() free function") {
        auto s = shape(4, 5, 6);
        CHECK(s == Shape<3, int>(4, 5, 6));
    }
}

TEST_CASE("Shape::fill/one/zero") {
    CHECK(Shape<3, int>::fill(7) == Shape<3, int>(7, 7, 7));
    CHECK(Shape<3, int>::one() == Shape<3, int>(1, 1, 1));
    CHECK(Shape<3, int>::zero() == Shape<3, int>(0, 0, 0));
}

TEST_CASE("Shape::from") {
    SECTION("same dimensionality copies values") {
        Vec<int, 3> v {4, 5, 6};
        CHECK(Shape<3, int>::from(v) == Shape<3, int>(4, 5, 6));
    }

    SECTION("growing dimensionality pads with one") {
        Vec<int, 2> v {4, 5};
        CHECK(Shape<4, int>::from(v) == Shape<4, int>(4, 5, 1, 1));
    }

    SECTION("shrinking dimensionality truncates") {
        Vec<int, 4> v {4, 5, 6, 7};
        CHECK(Shape<2, int>::from(v) == Shape<2, int>(4, 5));
    }
}

TEST_CASE("Shape converting constructor") {
    SECTION("growing dimensionality pads with one") {
        Shape<2, int> src(4, 5);
        Shape<4, int> dst(src);
        CHECK(dst == Shape<4, int>(4, 5, 1, 1));
    }

    SECTION("shrinking dimensionality succeeds when dropped axes are one") {
        Shape<3, int> src(4, 5, 1);
        Shape<2, int> dst(src);
        CHECK(dst == Shape<2, int>(4, 5));
    }

    SECTION("shrinking dimensionality throws when a dropped axis is not one") {
        Shape<3, int> src(4, 5, 2);
        CHECK_THROWS_AS((Shape<2, int>(src)), std::overflow_error);
    }

    SECTION("narrowing type that overflows throws") {
        Shape<2, int> src(300, 1);
        CHECK_THROWS_AS((Shape<2, signed char>(src)), std::overflow_error);
    }

    SECTION("narrowing type that fits succeeds") {
        Shape<2, int> src(4, 5);
        Shape<2, signed char> dst(src);
        CHECK(dst[0] == 4);
        CHECK(dst[1] == 5);
    }
}

TEST_CASE("Shape::is_convertible_to") {
    CHECK(Shape<2, int>(4, 5).is_convertible_to<2, long>());
    CHECK_FALSE(Shape<2, int>(300, 1).is_convertible_to<2, signed char>());
    CHECK(Shape<3, int>(4, 5, 1).is_convertible_to<2, int>());
    CHECK_FALSE(Shape<3, int>(4, 5, 2).is_convertible_to<2, int>());
}

TEST_CASE("Shape::get_or_default") {
    Shape<2, int> s(4, 5);

    CHECK(s.get_or_default(0) == 4);
    CHECK(s.get_or_default(1) == 5);
    CHECK(s.get_or_default(2) == 1);
    CHECK(s.get_or_default(2, 99) == 99);

    Shape<0, int> empty;
    CHECK(empty.get_or_default(0) == 1);
    CHECK(empty.get_or_default(0, 42) == 42);
}

TEST_CASE("Shape::is_empty") {
    CHECK_FALSE(Shape<2, int>(4, 5).is_empty());
    CHECK(Shape<2, int>(0, 5).is_empty());
    CHECK(Shape<2, int>(4, 0).is_empty());
    CHECK(Shape<2, int>(-1, 5).is_empty());
    CHECK_FALSE(Shape<0, int>().is_empty());
}

TEST_CASE("Shape::volume") {
    CHECK(Shape<3, int>(2, 3, 4).volume() == 24);
    CHECK(Shape<2, int>(0, 5).volume() == 0);
    CHECK(Shape<0, int>().volume() == 1);
    CHECK(Shape<1, int>(7).volume() == 7);
}

TEST_CASE("Shape::contains") {
    Shape<2, int> s(3, 4);

    SECTION("matching dimensionality") {
        CHECK(s.contains(Point<2, int>(0, 0)));
        CHECK(s.contains(Point<2, int>(2, 3)));
        CHECK_FALSE(s.contains(Point<2, int>(3, 0)));
        CHECK_FALSE(s.contains(Point<2, int>(0, 4)));
        CHECK_FALSE(s.contains(Point<2, int>(-1, 0)));
    }

    SECTION("point has more dims: extra dims must be zero") {
        CHECK(s.contains(Point<3, int>(1, 2, 0)));
        CHECK_FALSE(s.contains(Point<3, int>(1, 2, 1)));
    }

    SECTION("shape has more dims: missing point dims default to index 0") {
        Shape<3, int> s3(3, 4, 5);
        CHECK(s3.contains(Point<2, int>(1, 2)));

        Shape<3, int> s3_empty(3, 4, 0);
        CHECK_FALSE(s3_empty.contains(Point<2, int>(1, 2)));
    }
}

TEST_CASE("Shape equality") {
    CHECK(Shape<2, int>(4, 5) == Shape<2, int>(4, 5));
    CHECK(Shape<2, int>(4, 5) != Shape<2, int>(4, 6));

    SECTION("shapes of different dimensionality compare via one-padding") {
        CHECK(Shape<2, int>(4, 5) == Shape<3, int>(4, 5, 1));
        CHECK(Shape<2, int>(4, 5) != Shape<3, int>(4, 5, 2));
    }
}

TEST_CASE("concat(Shape, Shape)") {
    Shape<2, int> a(2, 3);
    Shape<3, int> b(4, 5, 6);

    Shape<5, int> c = concat(a, b);
    CHECK(c == Shape<5, int>(2, 3, 4, 5, 6));
}
