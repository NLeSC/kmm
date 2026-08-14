#include <vector>

#include "catch2/catch_all.hpp"

#include "kmm/runtime/memops/copy.hpp"

using namespace kmm;

TEST_CASE("copy (CPU)") {
    std::vector<int> src = {1, 2, 3, 4, 5, 6};
    std::vector<int> dst(6, 0);

    CopyDescription description;
    description.element_size = sizeof(int);
    description.add_dimension(6, sizeof(int), sizeof(int));

    copy(src.data(), dst.data(), description);

    CHECK(dst == src);
}

TEST_CASE("CopyDescription::src_range/dst_range") {
    SECTION("no dimensions") {
        CopyDescription description;
        description.element_size = sizeof(int);
        description.src_offset = 4;
        description.dst_offset = 8;

        CHECK(description.src_range() == Range<ptrdiff_t>(4, 4 + sizeof(int)));
        CHECK(description.dst_range() == Range<ptrdiff_t>(8, 8 + sizeof(int)));
    }

    SECTION("positive strides") {
        CopyDescription description;
        description.element_size = sizeof(int);
        description.add_dimension(4, sizeof(int), 2 * sizeof(int));

        CHECK(description.src_range() == Range<ptrdiff_t>(0, 4 * sizeof(int)));
        CHECK(description.dst_range() == Range<ptrdiff_t>(0, 3 * 2 * sizeof(int) + sizeof(int)));
    }

    SECTION("negative stride") {
        CopyDescription description;
        description.element_size = sizeof(int);
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
    FakeLayout<2> dst {/* offset */ 3, /* extents */ {2, 3}, /* strides */ {3, 1}};
    FakeLayout<2> src {/* offset */ 5, /* extents */ {2, 3}, /* strides */ {6, 2}};

    CopyDescription description = make_copy_description(dst, src, sizeof(int));

    CHECK(description.element_size == sizeof(int));
    CHECK(description.dst_offset == 3 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.src_offset == 5 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.num_dims == 2);

    CHECK(description.dims[0].extent == 2);
    CHECK(description.dims[0].dst_stride == 3 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.dims[0].src_stride == 6 * static_cast<ptrdiff_t>(sizeof(int)));

    CHECK(description.dims[1].extent == 3);
    CHECK(description.dims[1].dst_stride == 1 * static_cast<ptrdiff_t>(sizeof(int)));
    CHECK(description.dims[1].src_stride == 2 * static_cast<ptrdiff_t>(sizeof(int)));
}
