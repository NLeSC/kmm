#include "catch2/catch_all.hpp"

#include "kmm/utils/function_ref.hpp"

using namespace kmm;

static int add_one(int x) {
    return x + 1;
}

TEST_CASE("FunctionRef with lambda") {
    int counter = 0;
    auto counting = [&counter](int x) mutable {
        counter++;
        return x + counter;
    };

    function_ref<int(int)> ref = counting;
    CHECK(ref(1) == 2);
    CHECK(ref(1) == 3);
    CHECK(counter == 2);
}

TEST_CASE("FunctionRef with function pointer") {
    function_ref<int(int)> ref = add_one;
    CHECK(ref(41) == 42);
}

TEST_CASE("FunctionRef default state is null") {
    function_ref<int(int)> ref;
    CHECK(!ref);
}