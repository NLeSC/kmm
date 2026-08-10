#include <array>
#include <type_traits>

#include "catch2/catch_all.hpp"

#include "kmm/core/type_utils.hpp"

#define CHECK_TYPE(V, ...) CHECK(std::is_same<decltype(V), __VA_ARGS__>::value)

using namespace kmm;

TEST_CASE("IndexSequence methods") {
    CHECK(IndexSequence<0, 1, 2>::all([](auto i) { return i() < 3; }));
    CHECK_FALSE(IndexSequence<0, 1, 2>::all([](auto i) { return i() < 2; }));

    int sum = 0;
    IndexSequence<0, 1, 2, 3>::for_each<void>([&](auto i) { sum += static_cast<int>(i()); });
    CHECK(sum == 6);

    auto v = IndexSequence<0, 1, 2>::construct<std::array<int, 3>>([](auto i) {
        return static_cast<int>(i()) * 10;
    });

    CHECK(v[0] == 0);
    CHECK(v[1] == 10);
    CHECK(v[2] == 20);

    auto va = IndexSequence<0, 1, 2>::fill<std::array<int, 3>>(7);
    CHECK(va[0] == 7);
    CHECK(va[1] == 7);
    CHECK(va[2] == 7);

    //    using Seq = IndexSequence<5, 2, 8>;
    //    CHECK(Seq::contains<5>);
    //    CHECK(Seq::contains<2>);
    //    CHECK(Seq::contains<8>);
    //    CHECK_FALSE(Seq::contains<0>);
    //
    //    CHECK(Seq::location_of<5> == 0);
    //    CHECK(Seq::location_of<2> == 1);
    //    CHECK(Seq::location_of<8> == 2);
}

TEST_CASE("make_index_sequence") {
    CHECK_TYPE(make_index_sequence<0>(), IndexSequence<>);
    CHECK_TYPE(make_index_sequence<1>(), IndexSequence<0>);
    CHECK_TYPE(make_index_sequence<2>(), IndexSequence<0, 1>);
    CHECK_TYPE(make_index_sequence<3>(), IndexSequence<0, 1, 2>);
    CHECK_TYPE(make_index_sequence<4>(), IndexSequence<0, 1, 2, 3>);
}

TEST_CASE("range_index_sequence_t") {
    CHECK_TYPE((range_index_sequence_t<2, 5>()), IndexSequence<2, 3, 4>);
    CHECK_TYPE((range_index_sequence_t<4, 4>()), IndexSequence<>);
}

TEST_CASE("reverse_index_sequence") {
    CHECK_TYPE((reverse_index_sequence<4>()), IndexSequence<3, 2, 1, 0>);
}

TEST_CASE("drop_index_sequence") {
    CHECK_TYPE((drop_index_sequence<4, 0>()), IndexSequence<1, 2, 3>);
    CHECK_TYPE((drop_index_sequence<4, 1>()), IndexSequence<0, 2, 3>);
    CHECK_TYPE((drop_index_sequence<4, 2>()), IndexSequence<0, 1, 3>);
    CHECK_TYPE((drop_index_sequence<4, 3>()), IndexSequence<0, 1, 2>);
}

TEST_CASE("is_partial_permutation") {
    CHECK((is_partial_permutation<IndexSequence<>, 0>));
    CHECK((is_partial_permutation<IndexSequence<0>, 1>));
    CHECK((is_partial_permutation<IndexSequence<0, 1, 2>, 3>));
    CHECK((is_partial_permutation<IndexSequence<2, 0, 1>, 3>));
    CHECK((is_partial_permutation<IndexSequence<3, 1>, 4>));  // partial: doesn't cover 0 or 2

    CHECK_FALSE((is_partial_permutation<IndexSequence<0, 0>, 1>));
    CHECK_FALSE((is_partial_permutation<IndexSequence<1, 2, 1>, 3>));
    CHECK_FALSE((is_partial_permutation<IndexSequence<0, 1, 2, 0>, 3>));

    // out of bounds for the given N, even though otherwise injective
    CHECK_FALSE((is_partial_permutation<IndexSequence<0, 3>, 3>));
}

TEST_CASE("is_permutation") {
    CHECK(is_permutation<IndexSequence<>>);
    CHECK(is_permutation<IndexSequence<0>>);
    CHECK(is_permutation<IndexSequence<0, 1, 2>>);
    CHECK(is_permutation<IndexSequence<2, 0, 1>>);

    CHECK_FALSE(is_permutation<IndexSequence<0, 0>>);  // repeated index
    CHECK_FALSE(is_permutation<IndexSequence<3, 1>>);  // injective, but doesn't cover 0..size-1
}
