#include "catch2/catch_all.hpp"

#include "kmm/core/layout.hpp"

using namespace kmm;

TEST_CASE("Layout basics") {
    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout layout(Shape<2, int>(4, 5), {5, 1});

    CHECK(TestLayout::rank == 2);
    CHECK(layout.domain() == Shape<2, int>(4, 5));
    CHECK(layout.mapping().get(ConstIndex<0>()) == 5);
}

TEST_CASE("Layout::extent/begin/end/origin") {
    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout layout(Shape<2, int>(4, 5), {5, 1});

    CHECK(layout.extent(0) == 4);
    CHECK(layout.extent(1) == 5);
    CHECK(layout.extent(5) == 1);  // out-of-range axis defaults to one

    CHECK(layout.begin(0) == 0);
    CHECK(layout.end(0) == 4);
    CHECK(layout.begin(1) == 0);
    CHECK(layout.end(1) == 5);
    CHECK(layout.origin(0) == 0);
    CHECK(layout.origin(1) == 0);

    CHECK(layout.begin() == Point<2, int>(0, 0));
    CHECK(layout.end() == Point<2, int>(4, 5));
}

TEST_CASE("Layout::shape/bounds/size/is_empty") {
    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout layout(Shape<2, int>(4, 5), {5, 1});

    CHECK(layout.shape() == Shape<2, int>(4, 5));
    CHECK(layout.bounds().shape() == Shape<2, int>(4, 5));
    CHECK(layout.size() == 20);
    CHECK_FALSE(layout.is_empty());

    SECTION("an axis of size zero makes the layout empty") {
        TestLayout empty_layout(Shape<2, int>(0, 5), {5, 1});
        CHECK(empty_layout.is_empty());
        CHECK(empty_layout.size() == 0);
    }
}

TEST_CASE("Layout::contains") {
    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout layout(Shape<2, int>(4, 5), {5, 1});

    CHECK(layout.contains(Point<2, int>(0, 0)));
    CHECK(layout.contains(Point<2, int>(3, 4)));
    CHECK_FALSE(layout.contains(Point<2, int>(4, 0)));
    CHECK_FALSE(layout.contains(Point<2, int>(0, 5)));
}

TEST_CASE("Layout::stride/strides") {
    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout layout(Shape<2, int>(4, 5), {5, 1});

    CHECK(layout.stride(0) == 5);
    CHECK(layout.stride(1) == 1);

    auto strides = layout.strides();
    CHECK(strides[0] == 5);
    CHECK(strides[1] == 1);
}

TEST_CASE("Layout::local_offset/offset") {
    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout layout(Shape<2, int>(4, 5), {5, 1});

    CHECK(layout.local_offset(Point<2, int>(0, 0)) == 0);
    CHECK(layout.local_offset(Point<2, int>(1, 2)) == 5 * 1 + 1 * 2);
    CHECK(layout.local_offset(Point<2, int>(3, 4)) == 5 * 3 + 1 * 4);

    SECTION("offset combines base_offset and local_offset") {
        using BoundsLayout = ::kmm::Layout<Bounds<2, int>, Strided>;
        BoundsLayout bl(Bounds<2, int>(Range<int>(2, 6), Range<int>(3, 8)), {5, 1});

        CHECK(bl.offset(Point<2, int>(2, 3)) == 0);  // offset at the domain's own origin is zero
        CHECK(bl.offset(Point<2, int>(3, 5)) == 7);
    }
}

TEST_CASE("Layout::is_contiguous") {
    SECTION("row-major strides are contiguous in row-major order, not column-major") {
        using TestLayout = ::kmm::Layout<Shape<3, int>, Strides<int, int, int>>;
        TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

        CHECK(layout.is_contiguous());
        CHECK(layout.is_contiguous(MemoryOrder::RowMajor));
        CHECK_FALSE(layout.is_contiguous(MemoryOrder::ColMajor));
    }

    SECTION("column-major layout is contiguous in column-major order, not row-major") {
        auto layout = make_layout<ColMajor>(Shape<3, int>(4, 5, 6));

        CHECK(layout.is_contiguous(MemoryOrder::ColMajor));
        CHECK_FALSE(layout.is_contiguous(MemoryOrder::RowMajor));
        CHECK_FALSE(layout.is_contiguous());  // default order is row-major
    }

    SECTION("padded strides are not contiguous") {
        auto layout = make_layout<RowMajorPadded<2>>(Shape<2, int>(5, 3));

        CHECK_FALSE(layout.is_contiguous());
    }

    SECTION("rank 0 layout is always contiguous") {
        auto layout = make_layout<RowMajor>(Shape<0, int>());

        CHECK(layout.is_contiguous());
        CHECK(layout.is_contiguous(MemoryOrder::ColMajor));
    }
}

TEST_CASE("Layout::is_mapping_from_policy") {
    auto layout = make_layout<RowMajor>(Shape<2, int>(4, 5));
    CHECK(layout.is_mapping_from_policy());
    CHECK_FALSE(layout.is_mapping_from_policy<ColMajor>());

    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout mismatched(Shape<2, int>(4, 5), TestLayout::mapping_type(1, 1));
    CHECK_FALSE(mismatched.is_mapping_from_policy());
}

TEST_CASE("Layout::with_domain/with_mapping") {
    using TestLayout = ::kmm::Layout<Shape<2, int>, Strided>;
    TestLayout layout(Shape<2, int>(4, 5), {5, 1});

    auto same_shape = layout.with_domain(Shape<2, int>(2, 3));
    CHECK(same_shape.size() == 6);

    auto new_strides = layout.with_mapping(Strides(100, 10));
    CHECK(new_strides.stride(0) == 100);
    CHECK(new_strides.stride(1) == 10);
    CHECK(new_strides.shape() == Shape<2, int>(4, 5));
}

TEST_CASE("Layout::zero_origin") {
    using TestLayout = ::kmm::Layout<Bounds<2, int>, Strided>;
    TestLayout layout(Bounds<2, int>(Range<int>(2, 6), Range<int>(3, 8)), {5, 1});
    CHECK(layout.base_offset() == -(2 * 5 + 3));

    auto zo = layout.zero_origin();
    CHECK(zo.begin() == Point<2, int>(0, 0));
    CHECK(zo.shape() == Shape<2, int>(4, 5));
    CHECK(zo.base_offset() == 0);
}

TEST_CASE("Layout::move_origin") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    auto moved = layout.move_origin(Point<3, int>(1, 2, 3));

    CHECK(moved.begin() == Point<3, int>(0, 0, 0));
    CHECK(moved.shape() == Shape<3, int>(4, 5, 6));
    CHECK(moved.size() == 120);
    CHECK(moved.stride(0) == 30);
    CHECK(moved.stride(1) == 6);
    CHECK(moved.stride(2) == 1);
    CHECK(moved.local_offset({1, 2, 3}) == 1 * 30 + 2 * 6 + 3 * 1);
    CHECK(moved.base_offset() == 1 * 30 + 2 * 6 + 3);
}

TEST_CASE("Layout::restrict_bounds") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    auto restricted =
        layout.restrict_bounds(Bounds<3, int>(Range<int>(1, 3), Range<int>(1, 4), Range<int>(0, 6))
        );

    CHECK(restricted.begin() == Point<3, int>(1, 1, 0));
    CHECK(restricted.end() == Point<3, int>(3, 4, 6));
    CHECK(restricted.size() == 2 * 3 * 6);
    CHECK(restricted.base_offset() == 0);
}

TEST_CASE("Layout::restrict_axis") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    auto restricted = layout.restrict_axis<0>(1, 3);

    CHECK(restricted.extent(0) == 2);
    CHECK(restricted.extent(1) == 5);
    CHECK(restricted.extent(2) == 6);
    CHECK(restricted.begin(0) == 1);
    CHECK(restricted.end(0) == 3);
    CHECK(restricted.base_offset() == 0);
}

TEST_CASE("Layout::slice_bounds") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    auto sliced =
        layout.slice_bounds(Bounds<3, int>(Range<int>(1, 3), Range<int>(1, 4), Range<int>(0, 6)));

    CHECK(sliced.begin() == Point<3, int>(0, 0, 0));
    CHECK(sliced.extent(0) == 2);
    CHECK(sliced.extent(1) == 3);
    CHECK(sliced.extent(2) == 6);
    CHECK(sliced.base_offset() == 1 * 30 + 1 * 6);
}

TEST_CASE("Layout::drop_axis") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    SECTION("dropping the middle axis keeps the outer two, in order") {
        auto dropped = layout.drop_axis<1>(2);

        CHECK(dropped.extent(0) == 4);
        CHECK(dropped.extent(1) == 6);
        CHECK(dropped.size() == 24);
        CHECK(dropped.stride(0) == 30);
        CHECK(dropped.stride(1) == 1);
        CHECK(dropped.base_offset() == 2 * 6);
    }

    SECTION("dropping the first axis") {
        auto dropped = layout.drop_axis<0>(0);

        CHECK(dropped.extent(0) == 5);
        CHECK(dropped.extent(1) == 6);
        CHECK(dropped.stride(0) == 6);
        CHECK(dropped.stride(1) == 1);
        CHECK(dropped.base_offset() == 0);
    }

    SECTION("dropping the last axis") {
        auto dropped = layout.drop_axis<2>(0);

        CHECK(dropped.extent(0) == 4);
        CHECK(dropped.extent(1) == 5);
        CHECK(dropped.stride(0) == 30);
        CHECK(dropped.stride(1) == 6);
        CHECK(dropped.base_offset() == 0);
    }
}

TEST_CASE("Layout::insert_axis") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    SECTION("inserting in the middle shifts later axes and gets stride zero") {
        auto inserted = layout.insert_axis<1>(7);

        CHECK(inserted.rank == 4);
        CHECK(inserted.extent(0) == 4);
        CHECK(inserted.extent(1) == 7);
        CHECK(inserted.extent(2) == 5);
        CHECK(inserted.extent(3) == 6);
        CHECK(inserted.stride(0) == 30);
        CHECK(inserted.stride(1) == 0);
        CHECK(inserted.stride(2) == 6);
        CHECK(inserted.stride(3) == 1);
        CHECK(inserted.size() == 4 * 7 * 5 * 6);
        CHECK(inserted.base_offset() == 0);
    }

    SECTION("inserting at the front") {
        auto inserted = layout.insert_axis<0>(2);

        CHECK(inserted.extent(0) == 2);
        CHECK(inserted.extent(1) == 4);
        CHECK(inserted.extent(2) == 5);
        CHECK(inserted.extent(3) == 6);
        CHECK(inserted.stride(0) == 0);
        CHECK(inserted.stride(1) == 30);
    }

    SECTION("inserting at the end") {
        auto inserted = layout.insert_axis<3>(2);

        CHECK(inserted.extent(3) == 2);
        CHECK(inserted.stride(0) == 30);
        CHECK(inserted.stride(3) == 0);
    }

    SECTION("default extent is one") {
        auto inserted = layout.insert_axis<0>();

        CHECK(inserted.extent(0) == 1);
        CHECK(inserted.stride(0) == 0);
    }

    SECTION("index along the broadcast axis does not affect the linear offset") {
        auto inserted = layout.insert_axis<1>(7);

        CHECK(inserted.local_offset({1, 0, 2, 3}) == 1 * 30 + 2 * 6 + 3 * 1);
        CHECK(inserted.local_offset({1, 5, 2, 3}) == 1 * 30 + 2 * 6 + 3 * 1);
    }
}

TEST_CASE("Layout::reverse_axes") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    auto reversed = layout.reverse_axes();

    CHECK(reversed.extent(0) == 6);
    CHECK(reversed.extent(1) == 5);
    CHECK(reversed.extent(2) == 4);
    CHECK(reversed.stride(0) == 1);
    CHECK(reversed.stride(1) == 6);
    CHECK(reversed.stride(2) == 30);
    CHECK(reversed.size() == 120);
    CHECK(reversed.base_offset() == 0);
}

TEST_CASE("Layout::slice_axis") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    SECTION("with a plain index drops the axis") {
        auto sliced = layout.slice_axis<1>(2);

        CHECK(sliced.extent(0) == 4);
        CHECK(sliced.extent(1) == 6);
        CHECK(sliced.stride(0) == 30);
        CHECK(sliced.stride(1) == 1);
        CHECK(sliced.base_offset() == 2 * 6);
    }

    SECTION("with a Range restricts the axis but keeps the rank") {
        auto sliced = layout.slice_axis<0>(Range<int>(1, 3));

        CHECK(sliced.extent(0) == 2);
        CHECK(sliced.extent(1) == 5);
        CHECK(sliced.extent(2) == 6);
        CHECK(sliced.begin(0) == 0);
        CHECK(sliced.local_offset({1, 0, 0}) == 1 * 30);
        CHECK(sliced.base_offset() == 1 * 30);
    }

    SECTION("with all leaves the axis unchanged") {
        auto sliced = layout.slice_axis<0>(all);

        CHECK(sliced.extent(0) == 4);
        CHECK(sliced.extent(1) == 5);
        CHECK(sliced.extent(2) == 6);
        CHECK(sliced.base_offset() == 0);
    }

    SECTION("with new_axis inserts a broadcast axis") {
        auto sliced = layout.slice_axis<1>(new_axis);

        CHECK(sliced.rank == 4);
        CHECK(sliced.extent(0) == 4);
        CHECK(sliced.extent(1) == 1);
        CHECK(sliced.extent(2) == 5);
        CHECK(sliced.extent(3) == 6);
        CHECK(sliced.stride(0) == 30);
        CHECK(sliced.stride(1) == 0);
        CHECK(sliced.stride(2) == 6);
        CHECK(sliced.stride(3) == 1);
        CHECK(sliced.base_offset() == 0);
    }
}

template<typename I>
__attribute__((noinline)) auto foobar(const I& input) {
    return input.slice(1, all, Range<int>(2, 4));
}

TEST_CASE("Layout::slice") {
    using TestLayout = ::kmm::Layout<Shape<3, int>, Strided>;
    TestLayout layout(Shape<3, int>(4, 5, 6), {30, 6, 1});

    SECTION("mixing an index, all, and a Range across axes") {
        foobar(layout);
        auto sliced = layout.slice(1, all, Range<int>(2, 4));

        CHECK(sliced.extent(0) == 5);
        CHECK(sliced.extent(1) == 2);
        CHECK(sliced.base_offset() == 1 * 30 + 1 * 2);
    }

    SECTION("all indices drops every axis") {
        auto sliced = layout.slice(1, 2, 3);

        CHECK(sliced.rank == 0);
        CHECK(sliced.size() == 1);
        CHECK(sliced.base_offset() == 1 * 30 + 2 * 6 + 3 * 1);
    }

    SECTION("all all_t leaves the layout unchanged") {
        auto sliced = layout.slice(all, all, all);

        CHECK(sliced.extent(0) == 4);
        CHECK(sliced.extent(1) == 5);
        CHECK(sliced.extent(2) == 6);
        CHECK(sliced.base_offset() == 0);
    }

    SECTION("mixing new_axis with other tokens") {
        auto sliced = layout.slice(all, new_axis, 2, all);

        CHECK(sliced.rank == 3);
        CHECK(sliced.extent(0) == 4);
        CHECK(sliced.extent(1) == 1);
        CHECK(sliced.extent(2) == 6);
        CHECK(sliced.stride(0) == 30);
        CHECK(sliced.stride(1) == 0);
        CHECK(sliced.stride(2) == 1);
        CHECK(sliced.base_offset() == 2 * 6);
    }
}

TEST_CASE("make_layout<RowMajor>") {
    SECTION("rank 3") {
        auto layout = make_layout<RowMajor>(Shape<3, int>(4, 5, 6));

        CHECK(layout.extent(0) == 4);
        CHECK(layout.extent(1) == 5);
        CHECK(layout.extent(2) == 6);
        CHECK(layout.stride(0) == 30);
        CHECK(layout.stride(1) == 6);
        CHECK(layout.stride(2) == 1);
        CHECK(layout.size() == 120);
        CHECK(layout.local_offset(Point<3, int>(1, 2, 3)) == 1 * 30 + 2 * 6 + 3 * 1);
    }

    SECTION("rank 4") {
        auto layout = make_layout<RowMajor>(Shape<4, int>(2, 3, 4, 5));

        CHECK(layout.stride(0) == 60);
        CHECK(layout.stride(1) == 20);
        CHECK(layout.stride(2) == 5);
        CHECK(layout.stride(3) == 1);
    }

    SECTION("rank 1") {
        auto layout = make_layout<RowMajor>(Shape<1, int>(7));

        CHECK(layout.stride(0) == 1);
        CHECK(layout.size() == 7);
    }

    SECTION("rank 0") {
        auto layout = make_layout<RowMajor>(Shape<0, int>());

        CHECK(layout.size() == 1);
    }

    SECTION("policy can be deduced from an argument instead of specified explicitly") {
        auto layout = make_layout(Shape<2, int>(4, 5), RowMajor {});

        CHECK(layout.stride(0) == 5);
        CHECK(layout.stride(1) == 1);
    }
}

TEST_CASE("make_layout<ColMajor>") {
    SECTION("rank 0") {
        auto layout = make_layout<ColMajor>(Shape<0, int>());

        CHECK(layout.size() == 1);
    }

    SECTION("rank 1") {
        auto layout = make_layout<ColMajor>(Shape<1, int>(7));

        CHECK(layout.stride(0) == 1);
        CHECK(layout.size() == 7);
    }

    SECTION("rank 3") {
        auto layout = make_layout<ColMajor>(Shape<3, int>(4, 5, 6));

        CHECK(layout.extent(0) == 4);
        CHECK(layout.extent(1) == 5);
        CHECK(layout.extent(2) == 6);
        CHECK(layout.stride(0) == 1);
        CHECK(layout.stride(1) == 4);
        CHECK(layout.stride(2) == 20);
        CHECK(layout.size() == 120);
        CHECK(layout.local_offset(Point<3, int>(1, 2, 3)) == 1 * 1 + 2 * 4 + 3 * 20);
    }

    SECTION("rank 4") {
        auto layout = make_layout<ColMajor>(Shape<4, int>(2, 3, 4, 5));

        CHECK(layout.stride(0) == 1);
        CHECK(layout.stride(1) == 2);
        CHECK(layout.stride(2) == 6);
        CHECK(layout.stride(3) == 24);
    }
}

TEST_CASE("make_layout<Strided>") {
    auto x = make_layout<Strided>(shape(5, 3), MemoryOrder::RowMajor);
    CHECK(x.strides() == Vec(3, 1));

    auto y = make_layout<Strided>(shape(5, 3), MemoryOrder::ColMajor);
    CHECK(y.strides() == Vec(1, 5));

    auto xp = make_layout<StridedPadded<2>>(shape(5, 3), MemoryOrder::RowMajor);
    CHECK(xp.strides() == Vec(4, 1));

    auto yp = make_layout<StridedPadded<2>>(shape(5, 3), MemoryOrder::ColMajor);
    CHECK(yp.strides() == Vec(1, 6));
}
