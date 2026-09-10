#include <stdexcept>

#include "catch2/catch_all.hpp"

#include "kmm/core/bounds.hpp"

using namespace kmm;

TEST_CASE("Bounds construction") {
    SECTION("default constructs all-empty ranges") {
        Bounds<2, int> b;
        CHECK(b[0] == Range<int>());
        CHECK(b[1] == Range<int>());
        CHECK(b.is_empty());
    }

    SECTION("variadic constructor takes exactly N ranges") {
        Bounds<2, int> b(Range<int>(0, 3), Range<int>(1, 4));
        CHECK(b[0] == Range<int>(0, 3));
        CHECK(b[1] == Range<int>(1, 4));
    }

    SECTION("bounds() free function") {
        auto b = bounds(Range<int>(0, 3), Range<int>(1, 4));
        CHECK(b[0] == Range<int>(0, 3));
        CHECK(b[1] == Range<int>(1, 4));
    }

    SECTION("wraps an existing Vec<Range<T>, N> unchanged") {
        Vec<Range<int>, 2> v {Range<int>(0, 3), Range<int>(1, 4)};
        Bounds<2, int> b(v);
        CHECK(b[0] == Range<int>(0, 3));
        CHECK(b[1] == Range<int>(1, 4));
    }

    SECTION("constructs from a Shape as the region [0, shape)") {
        Bounds<2, int> b(Shape<2, int>(3, 4));
        CHECK(b.begin() == Point<2, int>(0, 0));
        CHECK(b.end() == Point<2, int>(3, 4));
    }
}

TEST_CASE("Bounds::from_bounds") {
    Bounds<2, int> b = Bounds<2, int>::from_bounds(Point<2, int>(1, 2), Point<2, int>(4, 6));

    CHECK(b.begin() == Point<2, int>(1, 2));
    CHECK(b.end() == Point<2, int>(4, 6));
}

TEST_CASE("Bounds::from_offset_size") {
    Bounds<2, int> b = Bounds<2, int>::from_offset_size(Point<2, int>(1, 2), Shape<2, int>(3, 4));

    CHECK(b.begin() == Point<2, int>(1, 2));
    CHECK(b.end() == Point<2, int>(4, 6));
    CHECK(b.shape() == Shape<2, int>(3, 4));
}

TEST_CASE("Bounds::empty/one") {
    CHECK(Bounds<2, int>::empty().is_empty());
    CHECK_FALSE(Bounds<2, int>::one().is_empty());
    CHECK(Bounds<2, int>::one().shape() == Shape<2, int>(1, 1));
}

TEST_CASE("Bounds::from") {
    SECTION("same dimensionality copies ranges") {
        Vec<Range<int>, 2> v {Range<int>(0, 3), Range<int>(1, 4)};
        Bounds<2, int> b = Bounds<2, int>::from(v);
        CHECK(b[0] == Range<int>(0, 3));
        CHECK(b[1] == Range<int>(1, 4));
    }

    SECTION("growing dimensionality pads with the unit range 0...1") {
        Vec<Range<int>, 1> v {Range<int>(0, 3)};
        Bounds<2, int> b = Bounds<2, int>::from(v);
        CHECK(b[0] == Range<int>(0, 3));
        CHECK(b[1] == Range<int>(0, 1));
    }
}

TEST_CASE("Bounds converting constructor") {
    SECTION("growing dimensionality pads with the unit range") {
        Bounds<1, int> src(Range<int>(0, 3));
        Bounds<2, int> dst(src);
        CHECK(dst[0] == Range<int>(0, 3));
        CHECK(dst[1] == Range<int>(0, 1));
    }

    SECTION("shrinking dimensionality succeeds when dropped axes are the unit range") {
        Bounds<2, int> src(Range<int>(0, 3), Range<int>(0, 1));
        Bounds<1, int> dst(src);
        CHECK(dst[0] == Range<int>(0, 3));
    }

    SECTION("shrinking dimensionality throws when a dropped axis is not the unit range") {
        Bounds<2, int> src(Range<int>(0, 3), Range<int>(1, 4));
        CHECK_THROWS_AS((Bounds<1, int>(src)), std::overflow_error);
    }
}

TEST_CASE("Bounds::begin/end/size accessors") {
    Bounds<2, int> b(Range<int>(1, 4), Range<int>(2, 6));

    CHECK(b.begin(0) == 1);
    CHECK(b.end(0) == 4);
    CHECK(b.size(0) == 3);

    CHECK(b.begin(1) == 2);
    CHECK(b.end(1) == 6);
    CHECK(b.size(1) == 4);

    SECTION("out-of-range axis defaults to the unit range") {
        CHECK(b.begin(2) == 0);
        CHECK(b.end(2) == 1);
        CHECK(b.size(2) == 1);
    }
}

TEST_CASE("Bounds::is_empty") {
    CHECK_FALSE(Bounds<2, int>(Range<int>(0, 3), Range<int>(0, 4)).is_empty());
    CHECK(Bounds<2, int>(Range<int>(0, 0), Range<int>(0, 4)).is_empty());
    CHECK(Bounds<2, int>(Range<int>(3, 0), Range<int>(0, 4)).is_empty());
}

TEST_CASE("Bounds::volume") {
    CHECK(Bounds<2, int>(Range<int>(0, 3), Range<int>(0, 4)).volume() == 12);
    CHECK(Bounds<2, int>(Range<int>(0, 0), Range<int>(0, 4)).volume() == 0);
}

TEST_CASE("Bounds::intersection") {
    Bounds<1, int> a(Range<int>(0, 10));
    Bounds<1, int> b(Range<int>(5, 15));

    CHECK(a.intersection(b)[0] == Range<int>(5, 10));

    Bounds<1, int> c(Range<int>(20, 30));
    CHECK(a.intersection(c).is_empty());
}

TEST_CASE("Bounds::overlaps") {
    Bounds<1, int> a(Range<int>(0, 10));

    CHECK(a.overlaps(Bounds<1, int>(Range<int>(5, 15))));
    CHECK_FALSE(a.overlaps(Bounds<1, int>(Range<int>(10, 20))));
    CHECK_FALSE(a.overlaps(Bounds<1, int>(Range<int>(5, 5))));

    SECTION("overlaps(Shape)") {
        CHECK(a.overlaps(Shape<1, int>(5)));
        CHECK_FALSE(a.overlaps(Shape<1, int>(0)));
    }
}

TEST_CASE("Bounds::contains(Bounds)") {
    Bounds<1, int> outer(Range<int>(0, 10));

    CHECK(outer.contains(Bounds<1, int>(Range<int>(2, 8))));
    CHECK(outer.contains(Bounds<1, int>(Range<int>(0, 10))));
    CHECK_FALSE(outer.contains(Bounds<1, int>(Range<int>(-1, 8))));
    CHECK_FALSE(outer.contains(Bounds<1, int>(Range<int>(2, 11))));
    CHECK(outer.contains(Bounds<1, int>(Range<int>(5, 5))));  // empty range always contained

    SECTION("contains(Shape)") {
        CHECK(outer.contains(Shape<1, int>(10)));
        CHECK_FALSE(outer.contains(Shape<1, int>(11)));
    }
}

TEST_CASE("Bounds::contains(Point)") {
    Bounds<2, int> b(Range<int>(0, 3), Range<int>(0, 4));

    CHECK(b.contains(Point<2, int>(0, 0)));
    CHECK(b.contains(Point<2, int>(2, 3)));
    CHECK_FALSE(b.contains(Point<2, int>(3, 0)));
    CHECK_FALSE(b.contains(Point<2, int>(0, 4)));

    SECTION("variadic overload") {
        CHECK(b.contains(1, 1));
        CHECK_FALSE(b.contains(3, 3));
    }
}
