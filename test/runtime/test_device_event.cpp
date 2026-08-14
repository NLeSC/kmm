#include "catch2/catch_all.hpp"

#include "kmm/runtime/device_event.hpp"

using namespace kmm;

TEST_CASE("DeviceEvent null") {
    DeviceEvent null_event;
    CHECK(null_event.is_null());
    CHECK(null_event == DeviceEvent::null());

    DeviceEvent event {DeviceStreamId(3), 1};
    CHECK_FALSE(event.is_null());
    CHECK(event.stream() == DeviceStreamId(3));
}

TEST_CASE("DeviceEvent ordering on same stream") {
    auto stream = DeviceStreamId(0);
    DeviceEvent a {stream, 1};
    DeviceEvent b {stream, 2};

    CHECK(a < b);
    CHECK(a.precedes(b));
    CHECK_FALSE(b.precedes(a));
}

TEST_CASE("DeviceEventSet construction from a single event") {
    SECTION("non-null event is kept") {
        DeviceEvent event {DeviceStreamId(1), 1};
        DeviceEventSet set {event};

        CHECK_FALSE(set.is_empty());
        CHECK(set.contains(event));
    }

    SECTION("null event is dropped, not stored") {
        DeviceEventSet set {DeviceEvent::null()};

        CHECK(set.is_empty());

        for (const auto& e : set) {
            CHECK_FALSE(e.is_null());
        }
    }
}

TEST_CASE("DeviceEventSet::insert") {
    SECTION("null events are ignored") {
        DeviceEventSet set;
        set.insert(DeviceEvent::null());

        CHECK(set.is_empty());
    }

    SECTION("events on the same stream are collapsed to the latest") {
        auto stream = DeviceStreamId(2);
        DeviceEvent early {stream, 1};
        DeviceEvent late {stream, 5};

        DeviceEventSet set;
        set.insert(early);
        set.insert(late);

        CHECK(set.contains(late));
        CHECK(set.find(stream) == late);

        size_t count = 0;
        for (const auto& e : set) {
            (void)e;
            count++;
        }
        CHECK(count == 1);
    }

    SECTION("events on different streams are kept separately") {
        DeviceEvent a {DeviceStreamId(0), 1};
        DeviceEvent b {DeviceStreamId(1), 1};

        DeviceEventSet set;
        set.insert(a);
        set.insert(b);

        CHECK(set.contains(a));
        CHECK(set.contains(b));
    }
}

TEST_CASE("DeviceEventSet::insert of another set never introduces null events") {
    DeviceEventSet source;
    source.insert(DeviceEvent {DeviceStreamId(0), 1});

    DeviceEventSet dest;
    dest.insert(source);

    for (const auto& e : dest) {
        CHECK_FALSE(e.is_null());
    }
}
