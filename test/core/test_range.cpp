#include <stdexcept>
#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/core/range.hpp"

using namespace kmm;

TEST_CASE("Range construction") {
    SECTION("default") {
        Range<int> r;
        CHECK(r.start == 0);
        CHECK(r.stop == 0);
        CHECK(r.is_empty());
    }

    SECTION("single argument") {
        Range<int> r(5);
        CHECK(r.start == 0);
        CHECK(r.stop == 5);
        CHECK_FALSE(r.is_empty());
    }

    SECTION("two arguments") {
        Range<int> r(2, 5);
        CHECK(r.start == 2);
        CHECK(r.stop == 5);
    }

    SECTION("range functions") {
        CHECK(range(5) == Range<int>(0, 5));
        CHECK(range(2, 5) == Range<int>(2, 5));
    }

    SECTION("one() is the range 0...1") {
        CHECK(Range<int>::one() == Range<int>(0, 1));
    }
}

TEST_CASE("Range::is_empty") {
    CHECK_FALSE(Range<int>(2, 5).is_empty());
    CHECK(Range<int>(5, 5).is_empty());
    CHECK(Range<int>(5, 2).is_empty());
}

TEST_CASE("Range::size") {
    CHECK(Range<int>(2, 5).size() == 3);
    CHECK(Range<int>(5, 5).size() == 0);
    CHECK(Range<int>(5, 2).size() == 0);
}

TEST_CASE("Range iteration") {
    SECTION("valid range") {
        std::vector<int> seen;
        for (auto i : Range<int>(5, 10)) {
            seen.push_back(i);
        }

        CHECK(seen == std::vector<int> {5, 6, 7, 8, 9});
    }

    SECTION("empty range") {
        std::vector<int> seen;
        for (auto i : Range<int>(5, 5)) {
            seen.push_back(i);
        }

        CHECK(seen.empty());
    }

    SECTION("invalid range") {
        std::vector<int> seen;
        for (auto i : Range<int>(5, 2)) {
            seen.push_back(i);
        }

        CHECK(seen.empty());
    }
}

TEST_CASE("Range::contains(index)") {
    // valid range
    Range<int> r(5, 10);
    CHECK_FALSE(r.contains(4));
    CHECK(r.contains(5));
    CHECK(r.contains(7));
    CHECK(r.contains(9));
    CHECK_FALSE(r.contains(10));

    // invalid range
    Range<int> s(10, 5);
    CHECK_FALSE(s.contains(4));
    CHECK_FALSE(s.contains(5));
    CHECK_FALSE(s.contains(7));
    CHECK_FALSE(s.contains(9));
    CHECK_FALSE(r.contains(10));
}

TEST_CASE("Range::contains(Range)") {
    Range<int> r(5, 10);

    CHECK(r.contains(Range<int>(5, 10)));
    CHECK(r.contains(Range<int>(6, 9)));
    CHECK(r.contains(Range<int>(7, 7)));  // empty range is always contained
    CHECK_FALSE(r.contains(Range<int>(4, 10)));
    CHECK_FALSE(r.contains(Range<int>(5, 11)));
    CHECK_FALSE(r.contains(Range<int>(0, 20)));
}

TEST_CASE("Range::overlaps") {
    Range<int> r(5, 10);

    CHECK(r.overlaps(Range<int>(5, 10)));
    CHECK(r.overlaps(Range<int>(0, 6)));
    CHECK(r.overlaps(Range<int>(9, 20)));
    CHECK_FALSE(r.overlaps(Range<int>(0, 5)));
    CHECK_FALSE(r.overlaps(Range<int>(10, 20)));
    CHECK_FALSE(r.overlaps(Range<int>(5, 5)));  // empty range never overlaps
}

TEST_CASE("Range::intersection") {
    CHECK(Range<int>(0, 10).intersection(Range<int>(5, 15)) == Range<int>(5, 10));
    CHECK(Range<int>(0, 5).intersection(Range<int>(10, 15)).is_empty());
    CHECK(Range<int>(0, 10).intersection(Range<int>(2, 8)) == Range<int>(2, 8));
}

TEST_CASE("Range::split") {
    SECTION("expected") {
        Range<int> r(0, 10);
        auto [a, b] = r.split(4);
        CHECK(a == Range<int>(0, 4));
        CHECK(b == Range<int>(4, 10));
    }

    SECTION("mid < start") {
        Range<int> r(5, 10);
        auto [a, b] = r.split(0);
        CHECK(a == Range<int>(5, 5));
        CHECK(b == Range<int>(5, 10));
    }

    SECTION("mid > stop") {
        Range<int> r(5, 10);
        auto [a, b] = r.split(20);
        CHECK(a == Range<int>(5, 10));
        CHECK(b == Range<int>(10, 10));
    }
}

TEST_CASE("Range::operator==/!=") {
    CHECK(Range<int>(2, 5) == Range<int>(2, 5));
    CHECK(Range<int>(2, 5) != Range<int>(2, 6));
    CHECK(Range<int>(2, 5) != Range<int>(3, 5));
}

TEST_CASE("Range shift") {
    CHECK(Range<int>(2, 5) + 3 == Range<int>(5, 8));
    CHECK(3 + Range<int>(2, 5) == Range<int>(5, 8));
    CHECK(Range<int>(2, 5) - 1 == Range<int>(1, 4));
}

TEST_CASE("Range::from and converting constructor") {
    SECTION("wider") {
        Range<short> src(2, 5);
        Range<int> dst = Range<int>::from(src);
        CHECK(dst == Range<int>(2, 5));

        Range<int> dst2(src);
        CHECK(dst2 == Range<int>(2, 5));
    }

    SECTION("narrowing ok") {
        Range<int> src(2, 5);
        Range<short> dst(src);
        CHECK(dst == Range<short>(2, 5));
    }

    SECTION("narrowing fails") {
        Range<int> src(0, 100000);
        CHECK_THROWS(Range<short>(src));
    }
}

TEST_CASE("Range::is_convertible_to") {
    CHECK(Range<int>(2, 5).is_convertible_to<long>());
    CHECK(Range<int>(2, 5).is_convertible_to<short>());
    CHECK_FALSE(Range<int>(0, 100000).is_convertible_to<short>());
}
