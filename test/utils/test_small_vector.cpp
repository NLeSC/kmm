#include <iterator>
#include <sstream>

#include "catch2/catch_all.hpp"

#include "kmm/utils/small_vector.hpp"

using namespace kmm;

TEST_CASE("small_vector construction") {
    small_vector<int, 4> a;
    CHECK(a.size() == 0);
    CHECK(a.is_empty());
    CHECK(a.capacity() == 4);
    CHECK_FALSE(a.is_heap_allocated());

    small_vector<int, 4> b {1, 2, 3};
    CHECK(b.size() == 3);
    CHECK_FALSE(b.is_empty());
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
}

TEST_CASE("small_vector::push_back") {
    SECTION("within capacity") {
        small_vector<int, 4> a;

        a.push_back(1);
        a.push_back(2);
        a.push_back(3);

        CHECK(a.size() == 3);
        CHECK(a.capacity() == 4);
        CHECK_FALSE(a.is_heap_allocated());
        CHECK(a[0] == 1);
        CHECK(a[1] == 2);
        CHECK(a[2] == 3);
    }

    SECTION("beyond capacity") {
        small_vector<int, 2> a;

        a.push_back(1);
        a.push_back(2);
        CHECK_FALSE(a.is_heap_allocated());

        a.push_back(3);
        CHECK(a.is_heap_allocated());
        CHECK(a.size() == 3);
        CHECK(a.capacity() >= 3);

        CHECK(a[0] == 1);
        CHECK(a[1] == 2);
        CHECK(a[2] == 3);
    }
}

TEST_CASE("small_vector::try_push_back") {
    small_vector<int, 2> a;

    CHECK(a.try_push_back(1));
    CHECK(a.try_push_back(2));
    CHECK(a.try_push_back(3));

    CHECK(a.size() == 3);
    CHECK(a[2] == 3);
}

TEST_CASE("small_vector copy constructor") {
    small_vector<int, 4> a {1, 2, 3};
    small_vector<int, 4> b = a;

    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);

    // modifying the copy should not affect the original
    b[0] = 99;
    CHECK(a[0] == 1);
    CHECK(b[0] == 99);
}

TEST_CASE("small_vector copy constructor with different data type and inline size") {
    small_vector<int, 4> a {1, 2, 3};
    small_vector<long, 8> b = a;

    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
}

TEST_CASE("small_vector copy assignment") {
    small_vector<int, 4> a {1, 2, 3};
    small_vector<int, 4> b {9, 9};

    b = a;

    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);

    // self-assignment should be a no-op
    b = b;
    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
}

TEST_CASE("small_vector move constructor") {
    small_vector<int, 4> a {1, 2, 3};
    small_vector<int, 4> b = std::move(a);

    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
}

TEST_CASE("small_vector move constructor with heap allocation") {
    small_vector<int, 2> a {1, 2, 3, 4};
    CHECK(a.is_heap_allocated());

    small_vector<int, 2> b = std::move(a);

    CHECK(b.is_heap_allocated());
    CHECK(b.size() == 4);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
    CHECK(b[3] == 4);
}

TEST_CASE("small_vector move assignment") {
    small_vector<int, 4> a {1, 2, 3};
    small_vector<int, 4> b {9};

    b = std::move(a);

    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
}

TEST_CASE("small_vector::resize") {
    SECTION("within inline") {
        small_vector<int, 4> a {1, 2};
        a.resize(4);
        CHECK(a.size() == 4);

        a.resize(1);
        CHECK(a.size() == 1);
        CHECK(a[0] == 1);
    }

    SECTION("beyond inline") {
        small_vector<int, 2> a {1, 2};
        a.resize(10);

        CHECK(a.size() == 10);
        CHECK(a.is_heap_allocated());
        CHECK(a[0] == 1);
        CHECK(a[1] == 2);
    }
}

TEST_CASE("small_vector::truncate") {
    small_vector<int, 4> a {1, 2, 3, 4};

    a.truncate(2);
    CHECK(a.size() == 2);
    CHECK(a[0] == 1);
    CHECK(a[1] == 2);

    // truncating to a larger size than current size should be a no-op
    a.truncate(10);
    CHECK(a.size() == 2);
}

TEST_CASE("small_vector::clear") {
    small_vector<int, 4> a {1, 2, 3};
    a.clear();

    CHECK(a.size() == 0);
    CHECK(a.is_empty());
    CHECK(a.capacity() == 4);
}

TEST_CASE("small_vector::insert_all") {
    SECTION("iterator") {
        small_vector<int, 4> a {1, 2};
        int extra[] = {3, 4, 5};

        a.insert_all(std::begin(extra), std::end(extra));

        CHECK(a.size() == 5);
        CHECK(a.is_heap_allocated());
        for (size_t i = 0; i < 5; i++) {
            CHECK(a[i] == static_cast<int>(i + 1));
        }
    }

    SECTION("from small_vector") {
        small_vector<int, 4> a {1, 2};
        small_vector<int, 8> b {3, 4};

        a.insert_all(b);

        CHECK(a.size() == 4);
        CHECK(a[0] == 1);
        CHECK(a[1] == 2);
        CHECK(a[2] == 3);
        CHECK(a[3] == 4);
    }
}

TEST_CASE("small_vector iterator") {
    small_vector<int, 4> a {1, 2, 3};

    int sum = 0;
    for (int v : a) {
        sum += v;
    }

    CHECK(sum == 6);

    auto it = a.begin();
    CHECK(*it == 1);
    CHECK(a.end() - a.begin() == 3);
}

TEST_CASE("small_vector operator<<") {
    small_vector<int, 4> a {1, 2, 3};
    CHECK(fmt::to_string(a) == "{1, 2, 3}");

    small_vector<int, 4> empty;
    CHECK(fmt::to_string(empty) == "{}");
}