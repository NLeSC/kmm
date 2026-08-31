#pragma once

#include <algorithm>
#include <cstddef>

#include "kmm/runtime/memops/types.hpp"

namespace kmm {

/// Shared core of `CopyDescription::simplify`, `FillDescription::simplify`, and
/// `ReductionDescription::simplify`: collapses to a single empty axis if any input axis has
/// extent zero, otherwise drops axes with extent one (they are visited exactly once, so their
/// stride never contributes to addressing), sorts the remaining axes into descending stride
/// order using `less`, and merges adjacent axes whenever `try_merge` reports that the outer axis
/// simply repeats the inner axis's memory layout without gaps.
///
/// The extent-zero case is handled up front, rather than folded into `try_merge` like the
/// contiguous-merge case, for two reasons: there may be no preceding axis yet to fold it into
/// (an all-zero-then-nonzero input has no "outer" for the first axis), and `try_merge` only ever
/// looks at stride-adjacent neighbors, so it can't guarantee collapsing to a single axis when the
/// zero-extent axis isn't adjacent (in sorted order) to every other axis. Callers rely on that
/// single-axis guarantee to hit their fast paths for an empty operation instead of falling back
/// to their "unsupported layout" case.
///
/// `less` must be a strict weak ordering over `Dim` that sorts by descending "primary" stride
/// (with whatever tie-breaking a given description wants). `try_merge(outer, inner)` must fold
/// `inner` into `outer` and return `true` if they are contiguous, or return `false` (leaving
/// `outer` untouched) otherwise. `out` must have room for `num_dims` entries.
template<typename Dim, typename Less, typename TryMerge>
size_t simplify_dims(const Dim* dims, size_t num_dims, Dim* out, Less less, TryMerge try_merge) {
    size_t n = 0;

    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i].extent <= 0) {
            out[0] = dims[i];
            return 1;
        }

        if (dims[i].extent > 1) {
            out[n] = dims[i];
            n++;
        }
    }

    // bubble sort
    for (size_t i = 0; i < MEMOPS_MAX_DIMS; i++) {
        for (size_t j = 0; j < i; j++) {
            if (i < n && less(out[i], out[j])) {
                std::swap(out[i], out[j]);
            }
        }
    }

    // Compact in place: `num_out` never exceeds `i`, so writing `out[num_out]` never clobbers an
    // entry at or beyond `out[i]` before it has been read.
    size_t num_out = 0;

    for (size_t i = 0; i < n; i++) {
        if (num_out == 0 || !try_merge(out[num_out - 1], out[i])) {
            out[num_out] = out[i];
            num_out++;
        }
    }

    return num_out;
}

}  // namespace kmm
