#include <stdexcept>
#include <type_traits>

#include "catch2/catch_all.hpp"

#include "kmm/core/strides.hpp"

#define CHECK_TYPE(V, ...) CHECK(std::is_same<decltype(V), __VA_ARGS__>::value)

using namespace kmm;

TEST_CASE("Strides N=0") {
    Strides<> s = {};

    CHECK(Strides<>::rank == 0);
    CHECK(s[0] == 0);
    CHECK(s.get(ConstIndex<0>()) == 0);
    CHECK(s.linearize_offset({}) == 0);
    CHECK(fmt::to_string(s) == "{}");
}

TEST_CASE("Strides N=1") {
    Strides<int> s = {5};

    CHECK(decltype(s)::rank == 1);
    CHECK(s[0] == 5);
    CHECK(s[1] == 0);
    CHECK(s.get(ConstIndex<0>()) == 5);
    CHECK(s.get(ConstIndex<1>()) == 0);
    CHECK(s.linearize_offset({2}) == 2 * 5);
    CHECK(fmt::to_string(s) == "{5}");

    Strides<ConstValue<5>> t = {{}};

    CHECK(decltype(t)::rank == 1);
    CHECK(t[0] == 5);
    CHECK(t[1] == 0);
    CHECK(t.get(ConstIndex<0>()) == 5);
    CHECK(t.get(ConstIndex<1>()) == 0);
    CHECK(t.linearize_offset({2}) == 2 * 5);
    CHECK(fmt::to_string(t) == "{5}");
}

TEST_CASE("Strides N=2") {
    Strides<int, int> s = {5, 1};

    CHECK(decltype(s)::rank == 2);
    CHECK(s[0] == 5);
    CHECK(s[1] == 1);
    CHECK(s[2] == 0);
    CHECK(s.get(ConstIndex<0>()) == 5);
    CHECK(s.get(ConstIndex<1>()) == 1);
    CHECK(s.get(ConstIndex<2>()) == 0);
    CHECK(s.linearize_offset({2, 10}) == 5 * 2 + 10);
    CHECK(fmt::to_string(s) == "{5, 1}");

    Strides<int, ConstValue<1>> t = {5, {}};

    CHECK(decltype(t)::rank == 2);
    CHECK(t[0] == 5);
    CHECK(t[1] == 1);
    CHECK(t[2] == 0);
    CHECK(t.get(ConstIndex<0>()) == 5);
    CHECK(t.get(ConstIndex<1>()) == 1);
    CHECK(t.get(ConstIndex<2>()) == 0);
    CHECK(t.linearize_offset({2, 10}) == 5 * 2 + 10);
    CHECK(fmt::to_string(t) == "{5, 1}");

    Strides<ConstValue<5>, ConstValue<1>> p;

    CHECK(decltype(p)::rank == 2);
    CHECK(p[0] == 5);
    CHECK(p[1] == 1);
    CHECK(p[2] == 0);
    CHECK(p.get(ConstIndex<0>()) == 5);
    CHECK(p.get(ConstIndex<1>()) == 1);
    CHECK(p.get(ConstIndex<2>()) == 0);
    CHECK(p.linearize_offset({2, 10}) == 5 * 2 + 10);
    CHECK(fmt::to_string(p) == "{5, 1}");
}

TEST_CASE("Strides N=3") {
    Strides<int, int, int> s = {10, 5, 1};

    CHECK(decltype(s)::rank == 3);
    CHECK(s[0] == 10);
    CHECK(s[1] == 5);
    CHECK(s[2] == 1);
    CHECK(s[3] == 0);
    CHECK(s.get(ConstIndex<0>()) == 10);
    CHECK(s.get(ConstIndex<1>()) == 5);
    CHECK(s.get(ConstIndex<2>()) == 1);
    CHECK(s.get(ConstIndex<3>()) == 0);
    CHECK(s.linearize_offset({6, 2, 10}) == 6 * 10 + 5 * 2 + 10);
    CHECK(fmt::to_string(s) == "{10, 5, 1}");
}

TEST_CASE("Strides sizeof") {
    // N = 0 is empty
    CHECK(std::is_empty_v<Strides<>>);
    CHECK(sizeof(Strides<>) == 1);

    // Non-empty means N strides
    CHECK(sizeof(Strides<long>) == sizeof(long));
    CHECK(sizeof(Strides<long, long>) == 2 * sizeof(long));
    CHECK(sizeof(Strides<long, long, long>) == 3 * sizeof(long));

    // Only static stride is always empty
    CHECK(sizeof(Strides<ConstValue<1>>) == 1);
    CHECK(sizeof(Strides<ConstValue<1>, ConstValue<2>>) == 1);
    CHECK(sizeof(Strides<ConstValue<1>, ConstValue<2>, ConstValue<3>>) == 1);
}

TEST_CASE("Strides default constructor") {
    SECTION("N=1") {
        Strides<int> s;
        CHECK(decltype(s)::rank == 1);
        CHECK(s[0] == 0);
    }

    SECTION("N=2") {
        Strides<int, int> s;
        CHECK(s[0] == 0);
        CHECK(s[1] == 0);
    }

    SECTION("N=2, static values") {
        Strides<int, ConstValue<5>> s;
        CHECK(s[0] == 0);
        CHECK(s[1] == 5);
    }
}

TEST_CASE("Strides converting constructor") {
    SECTION("dynamic to static, success") {
        Strides<int> src = {5};
        Strides<ConstValue<5>> dst = src;
        CHECK(dst[0] == 5);
    }

    SECTION("dynamic to static, failure") {
        Strides<int> src = {7};
        CHECK_THROWS_AS((Strides<ConstValue<5>>(src)), std::overflow_error);
    }

    SECTION("static to dynamic, success") {
        Strides<ConstValue<5>> src;
        Strides<int> dst = src;
        CHECK(dst[0] == 5);
    }

    SECTION("widening, success") {
        Strides<int> src = {7};
        Strides<long long> dst = src;
        CHECK(dst[0] == 7);
    }

    SECTION("narrowing, success") {
        Strides<int> src = {100};
        Strides<signed char> dst = src;
        CHECK(dst[0] == 100);
    }

    SECTION("narrowing, fails") {
        Strides<int> src = {300};
        CHECK_THROWS_AS((Strides<signed char>(src)), std::overflow_error);
    }
}

TEST_CASE("Strides::to_vec") {
    SECTION("rank 0") {
        Strides<> s;
        auto v = s.to_vec();
        CHECK_TYPE(v, Vec<default_stride_type, 0>);
    }

    SECTION("rank 2 with mixed dynamic/static axes") {
        Strides<int, ConstValue<1>> s = {5, {}};
        auto v = s.to_vec();

        CHECK_TYPE(v, Vec<default_stride_type, 2>);
        CHECK(v[0] == 5);
        CHECK(v[1] == 1);
    }
}

TEST_CASE("Strides::operator==/!=") {
    Strides<> a;
    Strides<int> b = {5};
    Strides<int, int> c = {0, 0};
    Strides<int, int> d = {5, 0};
    Strides<ConstValue<5>, ConstValue<1>> e;

    CHECK(a == a);
    CHECK(a != b);
    CHECK(a == c);
    CHECK(a != d);
    CHECK(a != e);

    CHECK(b != a);
    CHECK(b == b);
    CHECK(b != c);
    CHECK(b == d);
    CHECK(b != e);

    CHECK(c == a);
    CHECK(c != b);
    CHECK(c == c);
    CHECK(c != d);
    CHECK(c != e);

    CHECK(d != a);
    CHECK(d == b);
    CHECK(d != c);
    CHECK(d == d);
    CHECK(d != e);

    CHECK(e != a);
    CHECK(e != b);
    CHECK(e != c);
    CHECK(e != d);
    CHECK(e == e);

    // different data types
    Strides<int> f = {5};
    Strides<long long> g = {5};
    Strides<long long> h = {LONG_MAX};

    CHECK(f == g);
    CHECK(g == f);
    CHECK(f != h);
    CHECK(h != f);
}

TEST_CASE("make_strides") {
    auto x = make_strides(5, 2, ConstValue<1>());

    CHECK_TYPE(x, Strides<int, int, ConstValue<1>>);
    CHECK(x == Strides(5, 2, 1));

    auto y = make_strides(ConstValue<5>(), ConstValue<1>());

    CHECK_TYPE(y, Strides<ConstValue<5>, ConstValue<1>>);
    CHECK(y == Strides(5, 1));
}

TEST_CASE("make_strides_from_shape") {
    using index_t = default_index_type;

    SECTION("N=0") {
        auto s0 = make_strides_from_shape(Shape());
        CHECK_TYPE(s0, Strides<>);

        auto s0_col = make_strides_from_shape<MemoryOrder::ColMajor>(Shape());
        CHECK_TYPE(s0_col, Strides<>);
    }

    SECTION("N=1") {
        auto s1 = make_strides_from_shape(Shape(index_t(7)));
        CHECK_TYPE(s1, Strides<ConstValue<index_t {1}>>);
        CHECK(s1[0] == 1);

        auto s1_col = make_strides_from_shape<MemoryOrder::ColMajor>(Shape(index_t(7)));
        CHECK_TYPE(s1_col, Strides<ConstValue<index_t {1}>>);
        CHECK(s1_col[0] == 1);
    }

    SECTION("N=2") {
        auto s2 = make_strides_from_shape(Shape(10, 5));
        CHECK_TYPE(s2, Strides<index_t, ConstValue<index_t {1}>>);
        CHECK(s2[0] == 5);
        CHECK(s2[1] == 1);

        auto s2_col = make_strides_from_shape<MemoryOrder::ColMajor>(Shape(10, 5));
        CHECK_TYPE(s2_col, Strides<ConstValue<index_t {1}>, index_t>);
        CHECK(s2_col[0] == 1);
        CHECK(s2_col[1] == 10);
    }

    SECTION("N=3") {
        auto s3 = make_strides_from_shape(Shape(10, 5, 3));
        CHECK_TYPE(s3, Strides<index_t, index_t, ConstValue<index_t {1}>>);
        CHECK(s3[0] == 15);
        CHECK(s3[1] == 3);
        CHECK(s3[2] == 1);

        auto s3_col = make_strides_from_shape<MemoryOrder::ColMajor>(Shape(10, 5, 3));
        CHECK_TYPE(s3_col, Strides<ConstValue<index_t {1}>, index_t, index_t>);
        CHECK(s3_col[0] == 1);
        CHECK(s3_col[1] == 10);
        CHECK(s3_col[2] == 50);
    }

    SECTION("N=3, aligned") {
        index_t alignment = 4;

        auto s3 = make_strides_from_shape(Shape(10, 5, 3), alignment);
        CHECK_TYPE(s3, Strides<index_t, index_t, ConstValue<index_t {1}>>);
        CHECK(s3[0] == 20);
        CHECK(s3[1] == 4);
        CHECK(s3[2] == 1);

        auto s3_col = make_strides_from_shape<MemoryOrder::ColMajor>(Shape(10, 5, 3), alignment);
        CHECK_TYPE(s3_col, Strides<ConstValue<index_t {1}>, index_t, index_t>);
        CHECK(s3_col[0] == 1);
        CHECK(s3_col[1] == 12);
        CHECK(s3_col[2] == 60);
    }
}