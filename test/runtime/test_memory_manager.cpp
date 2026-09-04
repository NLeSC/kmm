#include "catch2/catch_all.hpp"

#include "kmm/runtime/memory_manager.hpp"

using namespace kmm;

class A: public reference_count<A> {};

class B: public A {};

TEST_CASE("MemoryManager") {
    //    auto x = MemoryManager {nullptr, DeviceStreamRegistry {}};
    //    auto y = x.create_buffer(BufferLayout::for_type<int>(), "test");
    //
    //    auto z = make_refcnt<B>();
    //    refcnt_ptr<A> a = z;
}