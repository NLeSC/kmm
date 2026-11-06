#include <string>

#include "catch2/catch_all.hpp"

#include "kmm/utils/lru_cache.hpp"

using namespace kmm;

TEST_CASE("lru_cache construction") {
    lru_cache<int, int> cache;
    CHECK(cache.size() == 0);
    CHECK(cache.is_empty());
}

TEST_CASE("lru_cache::insert and find") {
    lru_cache<int, std::string> cache;

    cache.insert(1, "one");
    cache.insert(2, "two");

    CHECK(cache.size() == 2);
    CHECK(cache.contains(1));
    CHECK(cache.contains(2));

    auto* value = cache.find(1);
    REQUIRE(value != nullptr);
    CHECK(*value == "one");

    CHECK(cache.find(3) == nullptr);
    CHECK_FALSE(cache.contains(3));
}

TEST_CASE("lru_cache::insert overwrites existing key") {
    lru_cache<int, std::string> cache;

    cache.insert(1, "one");
    cache.insert(1, "uno");

    CHECK(cache.size() == 1);
    CHECK(*cache.find(1) == "uno");
}

TEST_CASE("lru_cache::find marks entry as most recently used") {
    lru_cache<int, std::string> cache;

    cache.insert(1, "one");
    cache.insert(2, "two");

    // Touch 1 via find so that 2 becomes the least recently used entry.
    cache.find(1);

    CHECK(*cache.least_recently_used() == 2);
}

TEST_CASE("lru_cache::touch marks entry as most recently used") {
    lru_cache<int, std::string> cache;

    cache.insert(1, "one");
    cache.insert(2, "two");
    CHECK(*cache.least_recently_used() == 1);

    cache.touch(1);
    CHECK(*cache.least_recently_used() == 2);

    // touching a missing key is a no-op
    cache.touch(99);
    CHECK(*cache.least_recently_used() == 2);
}

TEST_CASE("lru_cache::least_recently_used") {
    lru_cache<int, std::string> cache;
    CHECK(cache.least_recently_used() == nullptr);

    cache.insert(1, "one");
    cache.insert(2, "two");

    REQUIRE(cache.least_recently_used() != nullptr);
    CHECK(*cache.least_recently_used() == 1);
}

TEST_CASE("lru_cache::remove") {
    lru_cache<int, std::string> cache;

    cache.insert(1, "one");
    cache.insert(2, "two");

    CHECK(cache.remove(1));
    CHECK_FALSE(cache.contains(1));
    CHECK(cache.size() == 1);

    CHECK_FALSE(cache.remove(1));
}

TEST_CASE("lru_cache::clear") {
    lru_cache<int, std::string> cache;

    cache.insert(1, "one");
    cache.insert(2, "two");
    cache.clear();

    CHECK(cache.is_empty());
    CHECK(cache.size() == 0);
    CHECK(cache.least_recently_used() == nullptr);
}
