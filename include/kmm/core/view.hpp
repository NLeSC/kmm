#pragma once

#include "kmm/utils/fixed_vector.hpp"

namespace kmm {
namespace views {

using default_index_type = signed long int;  // int64_t
using default_stride_type = signed int;  // int32_t

template<typename I, I... Dims>
struct static_domain {
    static constexpr size_t rank = sizeof...(Dims);
    using index_type = I;

    KMM_HOST_DEVICE
    static static_domain from_domain(const static_domain& domain) noexcept {
        return domain;
    }

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        return static_cast<index_type>(0);
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        index_type sizes[rank + 1] = {Dims..., 0};
        return axis < rank ? sizes[axis] : static_cast<index_type>(1);
    }
};

template<typename D, typename D::index_type... Offsets>
struct static_offset {
    static_assert(D::rank == sizeof...(Offsets), "Number of offsets must match rank of domain");

    static constexpr size_t rank = D::rank;
    using index_type = typename D::index_type;

    KMM_HOST_DEVICE
    constexpr static_offset(D inner = {}) : m_inner(inner) {}

    template<typename D2>
    KMM_HOST_DEVICE static_offset(const static_offset<D2, Offsets...>& domain) noexcept :
        m_inner(domain.inner_domain()) {}

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        index_type offsets[rank + 1] = {Offsets..., 0};
        return m_inner.offset(axis) + (axis < rank ? offsets[axis] : static_cast<index_type>(0));
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        return m_inner.size(axis);
    }

    KMM_HOST_DEVICE
    constexpr D inner_domain() const noexcept {
        return m_inner;
    }

  private:
    D m_inner;
};

template<size_t N, typename I = default_index_type>
struct dynamic_domain {
    static constexpr size_t rank = N;
    using index_type = I;

    KMM_HOST_DEVICE
    dynamic_domain(fixed_vector<index_type, rank> sizes = {}) noexcept : m_sizes(sizes) {}

    template<I... Dims>
    KMM_HOST_DEVICE dynamic_domain(static_domain<index_type, Dims...>) noexcept :
        dynamic_domain(fixed_vector<index_type, rank> {Dims...}) {}

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        return static_cast<index_type>(0);
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        return axis < rank ? m_sizes[axis] : static_cast<index_type>(1);
    }

  private:
    fixed_vector<index_type, rank> m_sizes;
};

template<size_t N, typename I = default_index_type>
struct dynamic_subdomain {
    static constexpr size_t rank = N;
    using index_type = I;

    KMM_HOST_DEVICE
    constexpr dynamic_subdomain() noexcept {
        for (size_t i = 0; i < rank; i++) {
            m_sizes[i] = 0;
            m_offsets[i] = 0;
        }
    }

    KMM_HOST_DEVICE
    constexpr dynamic_subdomain(fixed_vector<index_type, rank> sizes) noexcept {
        for (size_t i = 0; i < rank; i++) {
            m_offsets[i] = 0;
            m_sizes[i] = sizes[i];
        }
    }

    KMM_HOST_DEVICE
    constexpr dynamic_subdomain(
        fixed_vector<index_type, rank> offsets,
        fixed_vector<index_type, rank> sizes
    ) noexcept {
        for (size_t i = 0; i < rank; i++) {
            m_offsets[i] = offsets[i];
            m_sizes[i] = sizes[i];
        }
    }

    KMM_HOST_DEVICE
    constexpr dynamic_subdomain(const dynamic_domain<rank, index_type>& domain) noexcept {
        for (size_t i = 0; i < rank; i++) {
            m_offsets[i] = 0;
            m_sizes[i] = domain.size(i);
        }
    }

    template<I... Dims>
    KMM_HOST_DEVICE dynamic_subdomain(static_domain<index_type, Dims...> domain) noexcept :
        dynamic_subdomain(dynamic_domain<rank, index_type>(domain)) {}

    template<typename D, I... Offsets>
    KMM_HOST_DEVICE dynamic_subdomain(static_offset<D, Offsets...> domain) noexcept :
        dynamic_subdomain(domain.inner_domain()) {
        for (size_t i = 0; i < rank; i++) {
            m_offsets[i] = static_cast<index_type>(domain.offset(i));
            m_sizes[i] = static_cast<index_type>(domain.size(i));
        }
    }

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        return axis < rank ? m_offsets[axis] : static_cast<index_type>(0);
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        return axis < rank ? m_sizes[axis] : static_cast<index_type>(1);
    }

  private:
    fixed_vector<index_type, rank> m_offsets;
    fixed_vector<index_type, rank> m_sizes;
};

template<typename S = default_stride_type, S... Strides>
struct static_mapping {
    static constexpr size_t rank = sizeof...(Strides);
    using stride_type = S;

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        S strides[rank] = {Strides...};
        return axis < rank ? strides[axis] : static_cast<stride_type>(0);
    }
};

template<typename S>
struct static_mapping<S> {
    static constexpr size_t rank = 0;
    using stride_type = S;

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return static_cast<stride_type>(0);
    }
};

template<typename S = default_stride_type>
using linear_mapping = static_mapping<S, static_cast<S>(1)>;

template<size_t N, size_t ContAxis, typename S = default_stride_type>
struct contiguous_axis_mapping {
    static_assert(ContAxis < N, "Axis cannot exceed dimensionality");
    static constexpr size_t rank = N;
    using stride_type = S;

    KMM_HOST_DEVICE
    explicit constexpr contiguous_axis_mapping(fixed_vector<stride_type, rank> strides) noexcept {
        for (size_t i = 0; i < rank - 1; i++) {
            m_strides[i] = i < ContAxis ? strides[i] : strides[i + 1];
        }
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        if (axis < ContAxis) {
            return m_strides[axis];
        } else if (axis == ContAxis) {
            return static_cast<stride_type>(1);
        } else if (axis < rank) {
            return m_strides[axis - 1];
        } else {
            return static_cast<stride_type>(0);
        }
    }

  private:
    stride_type m_strides[rank - 1];
};

template<typename S>
struct contiguous_axis_mapping<1, 0, S> {
    static constexpr size_t rank = 1;
    using stride_type = S;

    KMM_HOST_DEVICE
    explicit constexpr contiguous_axis_mapping(fixed_vector<stride_type, 1> strides) noexcept {}

    KMM_HOST_DEVICE constexpr contiguous_axis_mapping(static_mapping<stride_type, 1> m) noexcept {}

    KMM_HOST_DEVICE
    operator static_mapping<stride_type, 1>() {
        return {};
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return axis == 0 ? static_cast<stride_type>(1) : static_cast<stride_type>(0);
    }
};

template<size_t N, typename S = default_stride_type>
struct strided_mapping {
    static constexpr size_t rank = N;
    using stride_type = S;

    template<typename M>
    KMM_HOST_DEVICE static constexpr strided_mapping from_mapping(const M& mapping) noexcept {
        fixed_vector<stride_type, rank> strides;

        for (size_t i = 0; i < N; i++) {
            strides[i] = static_cast<stride_type>(mapping.stride(i));
        }

        return {strides};
    }

    KMM_HOST_DEVICE
    constexpr strided_mapping() noexcept {
        for (size_t i = 0; i < N; i++) {
            m_strides[i] = static_cast<stride_type>(0);
        }
    }

    KMM_HOST_DEVICE
    constexpr strided_mapping(fixed_vector<stride_type, rank> strides) noexcept {
        for (size_t i = 0; i < N; i++) {
            m_strides[i] = strides[i];
        }
    }

    template<size_t ContAxis>
    KMM_HOST_DEVICE constexpr strided_mapping(contiguous_axis_mapping<N, ContAxis, stride_type> m
    ) noexcept :
        strided_mapping(from_mapping(m)) {}

    template<stride_type... Strides>
    KMM_HOST_DEVICE constexpr strided_mapping(static_mapping<stride_type, Strides...> m) noexcept :
        strided_mapping(from_mapping(m)) {}

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return axis < rank ? m_strides[axis] : static_cast<stride_type>(0);
    }

  private:
    fixed_vector<stride_type, rank> m_strides;
};

template<typename S>
struct strided_mapping<0, S> {
    static constexpr size_t rank = 0;
    using stride_type = S;

    KMM_HOST_DEVICE
    constexpr strided_mapping(fixed_vector<stride_type, 0> strides = {}) noexcept {}

    KMM_HOST_DEVICE constexpr strided_mapping(static_mapping<stride_type> m) noexcept {}

    KMM_HOST_DEVICE
    operator static_mapping<stride_type, 0>() {
        return {};
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return static_cast<stride_type>(0);
    }
};

namespace details {
template<size_t N, size_t ConstAxis, typename S>
struct select_contiguous_axis_mapping {
    using type = contiguous_axis_mapping<N, ConstAxis, S>;
};

template<size_t ConstAxis, typename S>
struct select_contiguous_axis_mapping<0, ConstAxis, S> {
    using type = strided_mapping<0, S>;
};
}  // namespace details

template<typename S = default_stride_type>
struct left_to_right_layout {
    template<typename D>
    using mapping_type = typename details::select_contiguous_axis_mapping<D::rank, 0, S>::type;

    template<typename D>
    KMM_HOST_DEVICE static mapping_type<D> from_domain(const D& domain) noexcept {
        fixed_vector<S, D::rank> strides;
        S stride = 1;

        for (size_t i = 0; i < D::rank; i++) {
            strides[i] = stride;
            stride *= static_cast<S>(domain.size(i));
        }

        return mapping_type<D>(strides);
    }
};

template<typename S = default_stride_type>
struct right_to_left_layout {
    template<typename D>
    using mapping_type =
        typename details::select_contiguous_axis_mapping<D::rank, D::rank - 1, S>::type;

    template<typename D>
    KMM_HOST_DEVICE static mapping_type<D> from_domain(const D& domain) noexcept {
        fixed_vector<S, D::rank> strides;
        S stride = 1;

        for (size_t i = D::rank; i > 0; i--) {
            strides[i - 1] = stride;
            stride *= static_cast<S>(domain.size(i - 1));
        }

        return mapping_type<D>(strides);
    }
};

template<typename S = default_stride_type>
struct strided_layout {
    template<typename D>
    using mapping_type = strided_mapping<D::rank, S>;

    template<typename D>
    KMM_HOST_DEVICE static mapping_type<D> from_domain(const D& domain) noexcept {
        return mapping_type<D>::from_mapping(right_to_left_layout<S>::from_domain(domain));
    }
};

template<typename S, S... Strides>
struct static_layout {
    template<typename D>
    using mapping_type = static_mapping<S, Strides...>;

    template<typename D>
    KMM_HOST_DEVICE static mapping_type<D> from_domain(const D& domain) noexcept {
        static_assert(sizeof...(Strides) == D::rank, "number of strides must match dimensionality");
        return {};
    }
};

using default_layout = right_to_left_layout<>;

template<size_t Axis, typename D>
struct drop_axis_domain {
    static_assert(Axis < D::rank);
    using index_type = typename D::index_type;
    using type = dynamic_subdomain<D::rank - 1, index_type>;

    KMM_HOST_DEVICE
    static type call(const D& domain) noexcept {
        fixed_vector<index_type, D::rank - 1> new_offsets;
        fixed_vector<index_type, D::rank - 1> new_sizes;
        size_t axis = Axis;

        for (size_t i = 0; i < D::rank - 1; i++) {
            new_offsets[i] = domain.offset(i < axis ? i : i + 1);
            new_sizes[i] = domain.size(i < axis ? i : i + 1);
        }

        return {new_offsets, new_sizes};
    }
};

template<size_t DropAxis, size_t N, typename I>
struct drop_axis_domain<DropAxis, dynamic_domain<N, I>> {
    static_assert(DropAxis < N);
    using index_type = I;
    using type = dynamic_domain<N - 1, I>;

    KMM_HOST_DEVICE
    static type call(const dynamic_domain<N, I>& domain) noexcept {
        fixed_vector<index_type, N - 1> new_sizes;

        for (size_t i = 0; i < N - 1; i++) {
            new_sizes[i] = domain.size(i < DropAxis ? i : i + 1);
        }

        return {new_sizes};
    }
};

template<size_t DropAxis, typename I, I... Dims>
struct drop_axis_domain<DropAxis, static_domain<I, Dims...>>:
    drop_axis_domain<DropAxis, dynamic_domain<sizeof...(Dims), I>> {};

template<size_t DropAxis, typename L, typename D>
struct drop_axis_layout {
    using old_mapping_type = typename L::template mapping_type<D>;
    using stride_type = typename old_mapping_type::stride_type;

    using new_mapping_type = strided_mapping<D::rank - 1, stride_type>;
    using type = strided_layout<stride_type>;

    KMM_HOST_DEVICE
    static new_mapping_type call(const old_mapping_type& mapping) noexcept {
        fixed_vector<stride_type, D::rank - 1> new_strides;

        for (size_t i = 0; i < D::rank - 1; i++) {
            new_strides[i] = mapping.stride(i < DropAxis ? i : i + 1);
        }

        return {new_strides};
    }
};

template<typename S, typename D>
struct drop_axis_layout<0, right_to_left_layout<S>, D> {
    using old_mapping_type = contiguous_axis_mapping<D::rank, D::rank - 1, S>;
    using stride_type = S;

    using new_mapping_type =
        typename details::select_contiguous_axis_mapping<D::rank - 1, D::rank - 2, S>::type;
    using type = right_to_left_layout<S>;

    KMM_HOST_DEVICE
    static new_mapping_type call(const old_mapping_type& mapping) noexcept {
        fixed_vector<stride_type, D::rank - 1> new_strides;

        for (size_t i = 0; i < D::rank - 1; i++) {
            new_strides[i] = mapping.stride(i + 1);
        }

        return new_mapping_type {new_strides};
    }
};

struct host_accessor {
    template<typename T>
    KMM_HOST_DEVICE T& dereference_pointer(T* ptr) const noexcept {
        return *ptr;
    }
};

struct device_accessor {
    template<typename T>
    KMM_HOST_DEVICE T& dereference_pointer(T* ptr) const {
#if __CUDA_ARCH__ or __HIP_DEVICE_COMPILE__
        return *ptr;
#else
        throw std::runtime_error("device data cannot be accessed on host");
#endif
    }
};

template<typename A, typename B>
struct convert_pointer;

template<typename T>
struct convert_pointer<T, T> {
    static KMM_HOST_DEVICE T* call(T* p) {
        return p;
    }
};

template<typename T>
struct convert_pointer<T, const T>: convert_pointer<const T, const T> {};

}  // namespace views
template<typename View, typename T, typename D, size_t K = 0, size_t N = D::rank>
struct ViewSubscript {
    using type = ViewSubscript;
    using subscript_type = typename ViewSubscript<View, T, D, K + 1>::type;
    using index_type = typename D::index_type;
    using ndindex_type = fixed_vector<index_type, D::rank>;

    KMM_HOST_DEVICE
    static type instantiate(const View* base, ndindex_type index = {}) noexcept {
        return type {base, index};
    }

    KMM_HOST_DEVICE
    ViewSubscript(const View* base, ndindex_type index) noexcept : base_(base), index_(index) {}

    KMM_HOST_DEVICE
    subscript_type operator[](index_type index) {
        index_[K] = index;
        return ViewSubscript<View, T, D, K + 1>::instantiate(base_, index_);
    }

  private:
    const View* base_;
    ndindex_type index_;
};

template<typename View, typename T, typename D, size_t N>
struct ViewSubscript<View, T, D, N, N> {
    using type = T&;
    using index_type = typename D::index_type;
    using ndindex_type = fixed_vector<index_type, N>;

    KMM_HOST_DEVICE
    static type instantiate(const View* base, ndindex_type index) {
        return base->access(index);
    }
};

template<typename Derived, typename T, typename D, size_t N = D::rank>
struct AbstractViewBase {
    using index_type = typename D::index_type;
    using subscript_type = typename ViewSubscript<Derived, T, D>::subscript_type;

    KMM_HOST_DEVICE
    subscript_type operator[](index_type index) const {
        return ViewSubscript<Derived, T, D>::instantiate(static_cast<const Derived*>(this))[index];
    }
};

template<typename Derived, typename T, typename D>
struct AbstractViewBase<Derived, T, D, 0> {
    using reference = T&;

    KMM_HOST_DEVICE
    reference operator*() const {
        return static_cast<const Derived*>(this)->access({});
    }
};

template<typename T, typename D, typename L, typename A = views::host_accessor>
struct AbstractView:
    public D,
    public L::template mapping_type<D>,
    public A,
    public AbstractViewBase<AbstractView<T, D, L, A>, T, D> {
    using self_type = AbstractView;
    using value_type = T;
    using domain_type = D;
    using layout_type = L;
    using mapping_type = typename L::template mapping_type<D>;
    using accessor_type = A;
    using pointer = T*;
    using reference = T&;

    static constexpr size_t rank = D::rank;
    using index_type = typename domain_type::index_type;
    using stride_type = typename mapping_type::stride_type;
    using ndindex_type = fixed_vector<index_type, rank>;
    using ndstride_type = fixed_vector<stride_type, rank>;

    using origin_domain_type = views::dynamic_domain<rank, index_type>;
    using shifted_domain_type = views::dynamic_subdomain<rank, index_type>;

    AbstractView(const AbstractView&) = default;
    AbstractView(AbstractView&&) noexcept = default;

    AbstractView& operator=(const AbstractView&) = default;
    AbstractView& operator=(AbstractView&&) noexcept = default;

    KMM_HOST_DEVICE
    AbstractView(
        pointer data,
        domain_type domain,
        mapping_type mapping,
        accessor_type accessor = {}
    ) noexcept :
        domain_type(domain),
        mapping_type(mapping),
        accessor_type(accessor) {
        m_data = data - this->linearize_index(offsets());
    }

    KMM_HOST_DEVICE
    AbstractView(pointer data = nullptr, domain_type domain = {}) noexcept :
        AbstractView(data, domain, layout_type::from_domain(domain)) {}

    template<
        typename T2,
        typename D2,
        typename L2,
        typename = decltype(views::convert_pointer<T2, T>::call(nullptr))>
    KMM_HOST_DEVICE AbstractView(const AbstractView<T2, D2, L2, A>& that) noexcept :
        AbstractView(
            views::convert_pointer<T2, T>::call(that.data()),
            domain_type(that.domain()),
            mapping_type(that.mapping()),
            that.accessor()
        ) {}

    template<typename T2, typename D2, typename L2>
    KMM_HOST_DEVICE AbstractView& operator=(const AbstractView<T2, D2, L2, A>& that) noexcept {
        return *this = AbstractView(that);
    }

    KMM_HOST_DEVICE
    pointer data() const noexcept {
        return data_at(offsets());
    }

    KMM_HOST_DEVICE
    operator pointer() const noexcept {
        return data();
    }

    KMM_HOST_DEVICE
    const mapping_type& mapping() const noexcept {
        return *this;
    }

    KMM_HOST_DEVICE
    const domain_type& domain() const noexcept {
        return *this;
    }

    KMM_HOST_DEVICE
    const accessor_type& accessor() const noexcept {
        return *this;
    }

    KMM_HOST_DEVICE
    index_type size(size_t axis) const noexcept {
        return domain().size(axis);
    }

    KMM_HOST_DEVICE
    index_type size() const noexcept {
        index_type volume = 1;
        for (size_t i = 0; i < rank; i++) {
            volume *= domain().size(i);
        }
        return volume;
    }

    KMM_HOST_DEVICE
    size_t size_in_bytes() const noexcept {
        size_t nbytes = sizeof(T);
        for (size_t i = 0; i < rank; i++) {
            nbytes *= static_cast<size_t>(domain().size(i));
        }
        return nbytes;
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis = 0) const noexcept {
        return mapping().stride(axis);
    }

    KMM_HOST_DEVICE
    index_type offset(size_t axis = 0) const noexcept {
        return domain().offset(axis);
    }

    KMM_HOST_DEVICE
    ndstride_type strides() const noexcept {
        ndstride_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = stride(i);
        }
        return result;
    }

    KMM_HOST_DEVICE
    ndindex_type offsets() const noexcept {
        ndindex_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = offset(i);
        }
        return result;
    }

    KMM_HOST_DEVICE
    ndindex_type sizes() const noexcept {
        ndindex_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = this->size(i);
        }
        return result;
    }

    KMM_HOST_DEVICE
    index_type begin(size_t axis = 0) const noexcept {
        return offset(axis);
    }

    KMM_HOST_DEVICE
    index_type end(size_t axis = 0) const noexcept {
        return begin(axis) + this->size(axis);
    }

    template<typename P = ptrdiff_t>
    KMM_HOST_DEVICE P linearize_index(ndindex_type ndindex, P base = {}) const noexcept {
        for (size_t i = 0; i < rank; i++) {
            base +=
                static_cast<ptrdiff_t>(ndindex[i]) * static_cast<ptrdiff_t>(mapping().stride(i));
        }

        return base;
    }

    KMM_HOST_DEVICE
    value_type* data_at(ndindex_type ndindex) const noexcept {
        return linearize_index(ndindex, m_data);
    }

    template<typename... Indices>
    KMM_HOST_DEVICE value_type* data_at(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank, "invalid number of indices");
        return data_at(ndindex_type {indices...});
    }

    KMM_HOST_DEVICE
    reference access(ndindex_type ndindex) const noexcept {
        return accessor().dereference_pointer(data_at(ndindex));
    }

    template<typename... Indices>
    KMM_HOST_DEVICE reference operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank, "invalid number of indices");
        return access(ndindex_type {indices...});
    }

    KMM_HOST_DEVICE
    bool is_empty() const noexcept {
        bool result = false;
        for (size_t i = 0; i < rank; i++) {
            result |= domain().size(i) <= static_cast<index_type>(0);
        }
        return result;
    }

    KMM_HOST_DEVICE
    bool in_bounds(ndindex_type ndindex) const noexcept {
        bool result = true;
        for (size_t i = 0; i < rank; i++) {
            result &= ndindex[i] >= domain().offset(i);
            result &= ndindex[i] - domain().offset(i) < domain().size(i);
        }
        return result;
    }

    template<typename... Indices>
    KMM_HOST_DEVICE bool in_bounds(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank, "invalid number of indices");
        return in_bounds(ndindex_type {indices...});
    }

    KMM_HOST_DEVICE
    bool is_contiguous() const noexcept {
        stride_type curr = 1;
        bool result = true;

        for (size_t i = 0; i < rank; i++) {
            result &= mapping().stride(rank - i - 1) == curr;
            curr *= static_cast<stride_type>(domain().size(rank - i - 1));
        }

        return result;
    }

    template<size_t Axis = 0>
    KMM_HOST_DEVICE AbstractView<
        value_type,
        typename views::drop_axis_domain<Axis, domain_type>::type,
        typename views::drop_axis_layout<Axis, layout_type, domain_type>::type,
        accessor_type>
    drop_axis(index_type index) const noexcept {
        static_assert(Axis < rank, "axis out of bounds");
        return AbstractView<
            value_type,
            typename views::drop_axis_domain<Axis, domain_type>::type,
            typename views::drop_axis_layout<Axis, layout_type, domain_type>::type,
            accessor_type> {
            data() - mapping().stride(Axis) * offset(Axis) + mapping().stride(Axis) * index,
            views::drop_axis_domain<Axis, domain_type>::call(domain()),
            views::drop_axis_layout<Axis, layout_type, domain_type>::call(mapping()),
            accessor()
        };
    }

    template<size_t Axis = 0>
    KMM_HOST_DEVICE AbstractView<
        value_type,
        typename views::drop_axis_domain<Axis, domain_type>::type,
        typename views::drop_axis_layout<Axis, layout_type, domain_type>::type,
        accessor_type>
    drop_axis() const noexcept {
        static_assert(Axis < rank, "axis out of bounds");
        return this->template drop_axis<Axis>(offset(Axis));
    }

    AbstractView<value_type, origin_domain_type, layout_type, accessor_type>  //
    shift_to_origin() const noexcept {
        auto new_domain = views::dynamic_domain<rank, index_type>(sizes());
        return {data(), new_domain, mapping(), accessor()};
    }

    AbstractView<value_type, shifted_domain_type, layout_type, accessor_type>  //
    shift_to(ndindex_type new_offsets) const noexcept {
        auto new_domain = views::dynamic_subdomain<rank, index_type>(new_offsets, sizes());
        return {data(), new_domain, mapping(), accessor()};
    }

    AbstractView<value_type, shifted_domain_type, layout_type, accessor_type>  //
    shift_by(ndindex_type amount) const noexcept {
        auto new_offsets = offsets();
        for (size_t i = 0; i < rank; i++) {
            new_offsets[i] += amount[i];
        }
        return shift_to(new_offsets);
    }

    template<size_t Axis>
    AbstractView<value_type, shifted_domain_type, layout_type, accessor_type>  //
    shift_axis_to(index_type new_offset) const noexcept {
        static_assert(Axis < rank, "axis out of bounds");
        auto new_offsets = offsets();
        new_offsets[Axis] = new_offset;
        return shift_to(new_offsets);
    }

    template<size_t Axis>
    AbstractView<value_type, shifted_domain_type, layout_type, accessor_type>  //
    shift_axis_by(index_type amount) const noexcept {
        return shift_axis_to<Axis>(offset(Axis) + amount);
    }

  private:
    pointer m_data;
};

template<
    typename T,
    size_t N = 1,
    typename L = views::default_layout,
    typename A = views::host_accessor>
using View = AbstractView<const T, views::dynamic_domain<N>, L, A>;

template<
    typename T,
    size_t N = 1,
    typename L = views::default_layout,
    typename A = views::host_accessor>
using ViewMut = AbstractView<T, views::dynamic_domain<N>, L, A>;

template<
    typename T,
    size_t N = 1,
    typename L = views::default_layout,
    typename A = views::host_accessor>
using Subview = AbstractView<const T, views::dynamic_subdomain<N>, L, A>;

template<
    typename T,
    size_t N = 1,
    typename L = views::default_layout,
    typename A = views::host_accessor>
using SubviewMut = AbstractView<T, views::dynamic_subdomain<N>, L, A>;

template<typename T, size_t N = 1, typename A = views::host_accessor>
using ViewStrided = View<T, N, views::strided_layout<>, A>;

template<typename T, size_t N = 1, typename A = views::host_accessor>
using ViewStridedMut = ViewMut<T, N, views::strided_layout<>, A>;

template<typename T, size_t N = 1, typename A = views::host_accessor>
using SubviewStrided = Subview<T, N, views::strided_layout<>, A>;

template<typename T, size_t N = 1, typename A = views::host_accessor>
using SubviewStridedMut = SubviewMut<T, N, views::strided_layout<>, A>;

template<typename T, size_t N = 1, typename L = views::default_layout>
using GPUView = View<T, N, L, views::device_accessor>;

template<typename T, size_t N = 1, typename L = views::default_layout>
using GPUViewMut = ViewMut<T, N, L, views::device_accessor>;

template<typename T, size_t N = 1>
using GPUViewStrided = ViewStrided<T, N, views::device_accessor>;

template<typename T, size_t N = 1>
using GPUViewStridedMut = ViewStridedMut<T, N, views::device_accessor>;

template<typename T, size_t N = 1, typename L = views::default_layout>
using GPUSubview = Subview<T, N, L, views::device_accessor>;

template<typename T, size_t N = 1, typename L = views::default_layout>
using GPUSubviewMut = SubviewMut<T, N, L, views::device_accessor>;

template<typename T, size_t N = 1>
using GPUSubviewStrided = SubviewStrided<T, N, views::device_accessor>;

template<typename T, size_t N = 1>
using GPUSubviewStridedMut = SubviewStridedMut<T, N, views::device_accessor>;

}  // namespace kmm
