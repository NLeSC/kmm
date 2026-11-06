#include <type_traits>

#include "catch2/catch_all.hpp"

#include "kmm/core/const_value.hpp"

#define CHECK_TYPE(V, ...) CHECK(std::is_same<decltype(V), __VA_ARGS__>::value)

using namespace kmm;

TEST_CASE("ConstValue") {
    ConstValue<5> v;

    CHECK(v.value == 5);
    CHECK(static_cast<int>(v) == 5);
    CHECK(v() == 5);

    SECTION("converting constructor from a compatible ConstValue") {
        ConstValue<5L> from_int(v);
        CHECK(from_int.value == 5);
    }

    SECTION("operators") {
        CHECK_TYPE(-v, ConstValue<-5>);
        CHECK_TYPE(+v, ConstValue<5>);

        ConstValue<2> a;
        ConstValue<3> b;

        CHECK_TYPE(a + b, ConstValue<5>);
        CHECK_TYPE(a - b, ConstValue<-1>);
        CHECK_TYPE(a * b, ConstValue<6>);
        CHECK_TYPE(b / a, ConstValue<1>);

        CHECK(a + int(b) == 5);
        CHECK(a - int(b) == -1);
        CHECK(a * int(b) == 6);
        CHECK(b / int(a) == 1);
    }
}
