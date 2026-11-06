#include "catch2/catch_all.hpp"

#include "kmm/core/key_value.hpp"

using namespace kmm;

TEST_CASE("KeyValue construction") {
    // default constructor
    KeyValue<int> empty;
    (void)empty;

    // value constructor
    KeyValue<int> kv(42, 100);
    CHECK(kv.key == 42);
    CHECK(kv.value == 100);
}

TEST_CASE("KeyValue::operator==/!=") {
    KeyValue<int> a(1, 10);
    KeyValue<int> b(1, 10);
    KeyValue<int> c(2, 10);
    KeyValue<int> d(1, 20);

    CHECK(a == b);
    CHECK_FALSE(a != b);

    CHECK(a != c);
    CHECK_FALSE(a == c);

    CHECK(a != d);
    CHECK_FALSE(a == d);
}

TEST_CASE("KeyValue::operator</<=/>=/>") {
    // low < high
    KeyValue<int> low_value(5, 1);
    KeyValue<int> high_value(1, 2);
    CHECK(low_value < high_value);
    CHECK(low_value <= high_value);
    CHECK(high_value > low_value);
    CHECK(high_value >= low_value);

    KeyValue<int> a(1, 10);
    KeyValue<int> b(2, 10);

    // ties are broken using key
    CHECK(a < b);
    CHECK(a <= b);
    CHECK(b > a);
    CHECK(b >= a);

    // two equal items
    CHECK_FALSE(a < a);
    CHECK_FALSE(a > a);
    CHECK(a <= a);
    CHECK(a >= a);
}
