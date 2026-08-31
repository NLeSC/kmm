#include "catch2/catch_all.hpp"

#include "kmm/core/view.hpp"

using namespace kmm;

TEST_CASE("NDView default construction") {
    ViewMut<int, 2> view;

    CHECK(view.data() == nullptr);
    CHECK(view.is_empty());
}

TEST_CASE("make_view/NDView basics") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));

    CHECK(view.shape() == Shape<2, int>(4, 5));
    CHECK(view.extent(0) == 4);
    CHECK(view.extent(1) == 5);
    CHECK(view.size() == 20);
    CHECK_FALSE(view.is_empty());
    CHECK(view.stride(0) == 5);
    CHECK(view.stride(1) == 1);
    CHECK(view.strides() == Vec<default_stride_type, 2>(5, 1));
    CHECK(view.data() == data);
    CHECK(view.is_contiguous());
    CHECK(view.is_contiguous(MemoryOrder::RowMajor));
    CHECK_FALSE(view.is_contiguous(MemoryOrder::ColMajor));

    SECTION("an axis of extent zero makes the view empty") {
        auto empty_view = make_view(data, Shape<2, int>(0, 5));
        CHECK(empty_view.is_empty());
        CHECK(empty_view.size() == 0);
    }
}

TEST_CASE("make_view with an explicit stride policy") {
    int data[6] = {0, 1, 2, 3, 4, 5};
    auto view = make_view<ColMajor>(data, Shape<2, int>(2, 3));

    CHECK(view.stride(0) == 1);
    CHECK(view.stride(1) == 2);
    CHECK(view.is_contiguous(MemoryOrder::ColMajor));
    CHECK_FALSE(view.is_contiguous(MemoryOrder::RowMajor));
}

TEST_CASE("NDView::access") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));

    CHECK(view(1, 2) == data[1 * 5 + 2]);
    CHECK(view[Vec<int, 2>(1, 2)] == data[1 * 5 + 2]);
    CHECK(view[1][2] == data[1 * 5 + 2]);
    CHECK(view.access(Vec<int, 2>(1, 2)) == data[1 * 5 + 2]);

    SECTION("writes propagate back to the underlying storage") {
        view(0, 0) = 42;
        CHECK(data[0] == 42);
    }

    SECTION("contains") {
        CHECK(view.contains(Vec<int, 2>(0, 0)));
        CHECK(view.contains(Vec<int, 2>(3, 4)));
        CHECK_FALSE(view.contains(Vec<int, 2>(4, 0)));
        CHECK_FALSE(view.contains(Vec<int, 2>(0, 5)));
    }
}

TEST_CASE("NDView::move_origin") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));

    SECTION("shifting the first axis") {
        auto moved = view.move_origin(Vec<int, 2>(1, 0));
        CHECK(moved.shape() == Shape<2, int>(4, 5));

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                CHECK(moved(i, j) == view(i + 1, j));
            }
        }
    }

    SECTION("shifting the second axis") {
        auto moved = view.move_origin(Vec<int, 2>(0, 1));

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                CHECK(moved(i, j) == view(i, j + 1));
            }
        }
    }
}

TEST_CASE("NDView::restrict_bounds/restrict_axis") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));

    SECTION("restrict_axis narrows one axis, keeping absolute indices") {
        auto restricted = view.restrict_axis<0>(1, 3);

        CHECK(restricted.extent(0) == 2);
        CHECK(restricted.extent(1) == 5);
        CHECK(restricted(1, 2) == view(1, 2));
        CHECK(restricted(2, 3) == view(2, 3));
    }

    SECTION("restrict_bounds narrows to the intersection with the given bounds") {
        auto restricted = view.restrict_bounds(Bounds<2, int>(Range<int>(1, 3), Range<int>(0, 5)));

        CHECK(restricted.extent(0) == 2);
        CHECK(restricted.extent(1) == 5);
        CHECK(restricted(1, 0) == view(1, 0));
        CHECK(restricted(2, 4) == view(2, 4));
    }
}

TEST_CASE("NDView::zero_origin") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));
    auto restricted = view.restrict_axis<0>(1, 3);
    auto zo = restricted.zero_origin();

    CHECK(zo.shape() == Shape<2, int>(2, 5));
    CHECK(zo(0, 0) == view(1, 0));
    CHECK(zo(1, 4) == view(2, 4));
}

TEST_CASE("NDView::drop_axis") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));
    auto dropped = view.drop_axis<1>(2);

    CHECK(dropped.shape() == Shape<1, int>(4));

    for (int i = 0; i < 4; i++) {
        CHECK(dropped(i) == view(i, 2));
    }
}

TEST_CASE("NDView::insert_axis") {
    int data[4] = {10, 20, 30, 40};
    auto view = make_view(data, Shape<1, int>(4));

    auto inserted = view.insert_axis<1>(3);
    CHECK(inserted.shape() == Shape<2, int>(4, 3));

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(inserted(i, j) == view(i));
        }
    }
}

TEST_CASE("NDView::reverse_axes") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));
    auto reversed = view.reverse_axes();

    CHECK(reversed.shape() == Shape<2, int>(5, 4));

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK(reversed(j, i) == view(i, j));
        }
    }
}

TEST_CASE("NDView::slice_axis") {
    int data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<2, int>(4, 5));

    SECTION("with a start/end pair narrows the axis and rebases it to zero") {
        auto sliced = view.slice_axis<0>(1, 3);
        CHECK(sliced.shape() == Shape<2, int>(2, 5));

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 5; j++) {
                CHECK(sliced(i, j) == view(i + 1, j));
            }
        }
    }

    SECTION("with a Range token narrows the axis the same way") {
        auto sliced = view.slice_axis<1>(Range<int>(2, 4));
        CHECK(sliced.shape() == Shape<2, int>(4, 2));

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                CHECK(sliced(i, j) == view(i, j + 2));
            }
        }
    }

    SECTION("with all leaves the axis unchanged") {
        auto sliced = view.slice_axis<0>(all);
        CHECK(sliced.shape() == view.shape());
    }
}

TEST_CASE("NDView::slice") {
    int data[24];
    for (int i = 0; i < 24; i++) {
        data[i] = i;
    }

    auto view = make_view(data, Shape<3, int>(2, 3, 4));

    SECTION("mixing a plain index, all, and a Range across axes") {
        auto sliced = view.slice(1, all, Range<int>(1, 3));
        CHECK(sliced.shape() == Shape<2, int>(3, 2));

        CHECK(sliced.layout().base_offset() == view.layout().base_offset() + 1 * 12 + 1);
        CHECK(sliced.data() == view.data());

        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 2; k++) {
                CHECK(sliced(j, k) == view(1, j, k + 1));
            }
        }
    }

    SECTION("an index for every axis drops down to a rank-0 view") {
        auto sliced = view.slice(1, 2, 3);
        CHECK(decltype(sliced)::rank == 0);
        CHECK(sliced.size() == 1);
        CHECK(sliced() == view(1, 2, 3));
    }
}

TEST_CASE("NDView converting constructor") {
    int data[6] = {0, 1, 2, 3, 4, 5};
    auto view = make_view(data, Shape<2, int>(2, 3));

    NDView<const int, decltype(view)::layout_type> const_view = view;

    CHECK(const_view.data() == data);
    CHECK(const_view(1, 2) == 5);
}
