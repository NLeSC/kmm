#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/copy.hpp"

using namespace kmm;

TEST_CASE("copy (CPU)") {
    std::vector<int> src = {1, 2, 3, 4, 5, 6};
    std::vector<int> dst(6, 0);

    CopyDescription description(sizeof(int));
    description.add_dimension(6, sizeof(int), sizeof(int));

    copy(src.data(), dst.data(), description);

    CHECK(dst == src);
}

TEST_CASE("CopyDescription::src_range/dst_range") {
    SECTION("no dimensions") {
        CopyDescription description(sizeof(int));
        description.src_offset = 4;
        description.dst_offset = 8;

        CHECK(description.src_range() == Range<ptrdiff_t>(4, 4 + sizeof(int)));
        CHECK(description.dst_range() == Range<ptrdiff_t>(8, 8 + sizeof(int)));
    }

    SECTION("positive strides") {
        CopyDescription description(sizeof(int));
        description.add_dimension(4, sizeof(int), 2 * sizeof(int));

        CHECK(description.src_range() == Range<ptrdiff_t>(0, 4 * sizeof(int)));
        CHECK(description.dst_range() == Range<ptrdiff_t>(0, 3 * 2 * sizeof(int) + sizeof(int)));
    }

    SECTION("negative stride") {
        CopyDescription description(sizeof(int));
        description.add_dimension(4, -static_cast<memops_stride_type>(sizeof(int)), sizeof(int));

        CHECK(
            description.src_range()
            == Range<ptrdiff_t>(-3 * static_cast<ptrdiff_t>(sizeof(int)), sizeof(int))
        );
        CHECK(description.dst_range() == Range<ptrdiff_t>(0, 4 * sizeof(int)));
    }
}

namespace {
// Minimal duck-typed stand-in for `kmm::Layout`, avoiding a dependency on `kmm/core/layout.hpp`.
template<size_t N>
struct FakeLayout {
    static constexpr size_t rank = N;

    ptrdiff_t offset = 0;
    ptrdiff_t extents[N];
    ptrdiff_t strides[N];
    ptrdiff_t origins[N] = {};

    ptrdiff_t base_offset() const {
        return offset;
    }

    ptrdiff_t extent(size_t axis) const {
        return extents[axis];
    }

    ptrdiff_t stride(size_t axis) const {
        return strides[axis];
    }

    ptrdiff_t begin(size_t axis) const {
        return origins[axis];
    }
};
}  // namespace

TEST_CASE("make_copy_description") {
    // Both `dst` and `src` are contiguous here, so `add_dimension` folds the two axes into one.
    FakeLayout<2> dst {/* offset */ 3, /* extents */ {2, 3}, /* strides */ {3, 1}};
    FakeLayout<2> src {/* offset */ 5, /* extents */ {2, 3}, /* strides */ {6, 2}};

    CopyDescription description = make_copy_description(dst, src, sizeof(int));

    CHECK(description.element_size == sizeof(int));
    CHECK(description.dst_offset == 3 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.src_offset == 5 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.num_dims == 1);

    CHECK(description.dims[0].extent == 6);
    CHECK(description.dims[0].dst_stride == 1 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.dims[0].src_stride == 2 * static_cast<ptrdiff_t>(sizeof(int)));
}

TEST_CASE("make_copy_description (non-contiguous)") {
    // `dst` is not contiguous (there is a gap of 1 element between rows), so no merge happens.
    FakeLayout<2> dst {/* offset */ 3, /* extents */ {2, 3}, /* strides */ {4, 1}};
    FakeLayout<2> src {/* offset */ 5, /* extents */ {2, 3}, /* strides */ {6, 2}};

    CopyDescription description = make_copy_description(dst, src, sizeof(int));

    CHECK(description.num_dims == 2);

    CHECK(description.dims[0].extent == 2);
    CHECK(description.dims[0].dst_stride == 4 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.dims[0].src_stride == 6 * static_cast<ptrdiff_t>(sizeof(int)));

    CHECK(description.dims[1].extent == 3);
    CHECK(description.dims[1].dst_stride == 1 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.dims[1].src_stride == 2 * static_cast<ptrdiff_t>(sizeof(int)));
}

TEST_CASE("CopyDescription::simplify sorts axes without merging") {
    // Axes are added out of order but are not contiguous with each other, so `simplify` must
    // sort them into descending stride order without dropping any of them.
    CopyDescription description(sizeof(int));
    description.add_dimension(4, 7 * sizeof(int), 7 * sizeof(int));
    description.add_dimension(2, 101 * sizeof(int), 101 * sizeof(int));
    description.add_dimension(3, 23 * sizeof(int), 23 * sizeof(int));

    CopyDescription result = description.simplify();

    REQUIRE(result.num_dims == 3);
    CHECK(result.dims[0].extent == 2);
    CHECK(result.dims[1].extent == 3);
    CHECK(result.dims[2].extent == 4);
}

TEST_CASE("CopyDescription::simplify sorts and merges contiguous axes") {
    // Axes are added out of order, and two of them (stride 6 and stride 2) are contiguous with
    // each other. `simplify` must sort them into descending stride order first, then merge the
    // two contiguous ones, leaving the unrelated third axis (stride 100) untouched.
    CopyDescription description(sizeof(int));
    description.add_dimension(3, 2 * sizeof(int), 2 * sizeof(int));
    description.add_dimension(2, 100 * sizeof(int), 100 * sizeof(int));
    description.add_dimension(4, 6 * sizeof(int), 6 * sizeof(int));

    CopyDescription result = description.simplify();

    REQUIRE(result.num_dims == 2);

    CHECK(result.dims[0].extent == 2);
    CHECK(result.dims[0].src_stride == 100 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(result.dims[0].dst_stride == 100 * static_cast<ptrdiff_t>(sizeof(int)));

    CHECK(result.dims[1].extent == 12);
    CHECK(result.dims[1].src_stride == 2 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(result.dims[1].dst_stride == 2 * static_cast<ptrdiff_t>(sizeof(int)));
}

TEST_CASE("CopyDescription::simplify drops axes with extent one") {
    CopyDescription description(sizeof(int));
    description.add_dimension(4, 5 * sizeof(int), 5 * sizeof(int));
    description.add_dimension(1, 999 * sizeof(int), 999 * sizeof(int));

    CopyDescription result = description.simplify();

    REQUIRE(result.num_dims == 1);
    CHECK(result.dims[0].extent == 4);
    CHECK(result.dims[0].src_stride == 5 * static_cast<ptrdiff_t>(sizeof(int)));
}
