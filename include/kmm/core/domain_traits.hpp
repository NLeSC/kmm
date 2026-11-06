#pragma once

namespace kmm::detail {

/// Customization point mapping a domain type (e.g. `Shape`, `Bounds`, `FShape`) to the interface
/// `Layout` needs from it: rank, index type, per-axis bounds/extent, and how the domain itself
/// changes under axis operations (`slice_axis`, `drop_axis`, `permute_axes`, `insert_axis`).
/// Specialized alongside each domain type, in that type's own header.
template<typename T>
struct domain_traits;

}  // namespace kmm::detail
