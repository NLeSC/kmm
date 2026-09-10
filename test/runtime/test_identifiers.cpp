#include "catch2/catch_all.hpp"

#include "kmm/runtime/identifiers.hpp"

using namespace kmm;

TEST_CASE("MemoryId::host and MemoryId::device") {
    MemoryId host = MemoryId::host();
    CHECK(host.is_host());
    CHECK_FALSE(host.is_device());

    MemoryId device = MemoryId::device(DeviceId(2));
    CHECK(device.is_device());
    CHECK_FALSE(device.is_host());
    CHECK(device.as_device() == DeviceId(2));
}

TEST_CASE("MemoryId from string: host aliases") {
    for (const std::string& name : {"host", "cpu", "HOST", "Cpu"}) {
        MemoryId id = name;
        CHECK(id.is_host());
    }
}

TEST_CASE("MemoryId from string: device aliases without index default to device 0") {
    for (const std::string& name : {"gpu", "cuda", "hip", "device", "GPU", "Cuda"}) {
        MemoryId id = name;
        CHECK(id.is_device());
        CHECK(id.as_device() == DeviceId(0));
    }
}

TEST_CASE("MemoryId from string: device aliases with explicit index") {
    CHECK(MemoryId("gpu:0").as_device() == DeviceId(0));
    CHECK(MemoryId("cuda:1").as_device() == DeviceId(1));
    CHECK(MemoryId("device:3").as_device() == DeviceId(3));
    CHECK(MemoryId("CUDA:2").as_device() == DeviceId(2));
}

TEST_CASE("MemoryId from string: invalid strings throw") {
    CHECK_THROWS_AS(MemoryId("host:0"), std::runtime_error);
    CHECK_THROWS_AS(MemoryId("gpu:"), std::runtime_error);
    CHECK_THROWS_AS(MemoryId("gpu:x"), std::runtime_error);
    CHECK_THROWS_AS(MemoryId("gpu:1x"), std::runtime_error);
    CHECK_THROWS_AS(MemoryId("gpu:-1"), std::runtime_error);
    CHECK_THROWS_AS(MemoryId(""), std::runtime_error);
    CHECK_THROWS_AS(MemoryId("nonsense"), std::runtime_error);
}
