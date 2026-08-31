#pragma once

#include <optional>
#include <string>
#include <utility>

#include "kmm/api/accumulator.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

template<typename T, size_t K = 1>
struct Reduce {
    const T& target;
    ReductionOp op;
    Shape<K> replication;
};

template<typename T, typename LayoutT, typename... Extents>
Reduce<NDArray<T, LayoutT>, sizeof...(Extents)> reduce(
    const NDArray<T, LayoutT>& array,
    ReductionOp op,
    Extents... extents
) {
    auto replication = Shape<sizeof...(Extents)> {checked_cast<default_index_type>(extents)...};

    return Reduce<NDArray<T, LayoutT>, sizeof...(Extents)> {array, op, replication};
}

template<typename T, typename LayoutT, typename... Extents>
Reduce<NDArray<T, LayoutT>, sizeof...(Extents)> reduce(
    const NDArray<T, LayoutT>& array,
    Extents... extents
) {
    return reduce(array, ReductionOp::Sum, extents...);
}

template<typename T, typename LayoutT, typename... Extents>
Reduce<NDAccumulator<T, LayoutT>, sizeof...(Extents)> reduce(
    const NDAccumulator<T, LayoutT>& accumulator,
    Extents... extents
) {
    auto replication = Shape<sizeof...(Extents)> {checked_cast<default_index_type>(extents)...};

    return Reduce<NDAccumulator<T, LayoutT>, sizeof...(Extents)> {
        accumulator,
        accumulator.op(),
        replication};
}

namespace detail {

template<typename LayoutT, size_t K>
class ReplicatedRegion {
  public:
    using index_type = typename LayoutT::index_type;
    static constexpr size_t rank = LayoutT::rank;
    static constexpr size_t replicated_rank = rank + K;
    using replicated_shape = Shape<replicated_rank, index_type>;
    using replicated_strides = StridesN<default_stride_type, replicated_rank>;
    using replicated_layout = Layout<replicated_shape, replicated_strides>;

    ReplicatedRegion(const LayoutT& layout, const Shape<K>& replication) :
        m_source(layout.normalize_offset()),
        m_replication(replication) {
        KMM_ASSERT(m_replication.volume() >= 1);

        auto span = m_source.offset_span();
        m_replica_span = checked_sub<size_t>(span.stop, span.start);
    }

    size_t replica_span() const {
        return m_replica_span;
    }

    size_t replica_count() const {
        return checked_cast<size_t>(m_replication.volume());
    }

    size_t element_count() const {
        return checked_mul(replica_span(), replica_count());
    }

    replicated_layout view() const {
        auto array_shape = m_source.shape();
        auto array_strides = m_source.strides();

        replicated_shape shape;
        Vec<default_stride_type, replicated_rank> strides;

        if constexpr (rank > 0) {
            for (size_t i = 0; i < rank; i++) {
                shape[i] = array_shape[i];
                strides[i] = static_cast<default_stride_type>(array_strides[i]);
            }
        }

        auto stride = static_cast<default_stride_type>(replica_span());
        for (size_t j = K; j-- > 0;) {
            shape[rank + j] = m_replication[j];
            strides[rank + j] = stride;
            stride *= static_cast<default_stride_type>(m_replication[j]);
        }

        auto mapping = build_strides(strides, make_index_sequence<replicated_rank>());
        return replicated_layout(shape, mapping, m_source.base_offset());
    }

    ReductionDescription fold(ReductionOp op, DataType dtype) const {
        auto elem_size = checked_cast<memops_stride_type>(data_type_size(dtype));
        auto base_offset = checked_mul<memops_stride_type>(m_source.base_offset(), elem_size);

        ReductionDescription description(dtype, op);
        description.input_offset = base_offset;
        description.output_offset = base_offset;

        if constexpr (rank > 0) {
            auto array_shape = m_source.shape();
            auto array_strides = m_source.strides();

            for (size_t i = 0; i < rank; i++) {
                auto stride_bytes = checked_mul<memops_stride_type>(array_strides[i], elem_size);
                description.add_dimension(
                    checked_cast<memops_extent_type>(array_shape[i]),
                    stride_bytes,
                    stride_bytes
                );
            }
        }

        description.reduction_extent = checked_cast<memops_extent_type>(replica_count());
        description.reduction_stride = checked_mul<memops_stride_type>(replica_span(), elem_size);
        return description;
    }

  private:
    template<size_t... Is>
    KMM_HOST_DEVICE static replicated_strides
    build_strides(const Vec<default_stride_type, replicated_rank>& values, IndexSequence<Is...>) {
        return replicated_strides {values[Is]...};
    }

    LayoutT m_source;
    Shape<K> m_replication;
    size_t m_replica_span;
};

template<typename T, typename LayoutT, size_t K>
class LaunchArgReduce {
  public:
    using index_type = typename LayoutT::index_type;
    using region_type = ReplicatedRegion<LayoutT, K>;
    using replicated_layout = typename region_type::replicated_layout;
    using resolve_type = NDView<T, replicated_layout>;

    LaunchArgReduce(
        Buffer buffer,
        LayoutT layout,
        ReductionOp op,
        Shape<K, index_type> replication
    ) :
        m_buffer(std::move(buffer)),
        m_region(layout, replication),
        m_op(op) {}

    void acquire(Runtime& runtime, ResourceRequest& requests, MemoryId memory_id) {
        m_memory_id = memory_id;

        m_scratch = runtime.create_buffer(
            BufferLayout::for_type<T>(m_region.element_count()),
            "replicated partial",
            reduction_identity(data_type_of<T>(), m_op)
        );

        m_index = requests.add(memory_id, *m_scratch, AccessMode::ReadWrite);
    }

    resolve_type resolve(Runtime& runtime, const ResourceGrant& grant) {
        auto accessor = grant.accessor(m_index);
        auto layout = m_region.view();
        return {static_cast<typename resolve_type::pointer>(accessor.address), layout};
    }

    void release(Runtime& runtime) {
        if (!m_scratch) {
            return;
        }

        runtime.submit_reduction(
            m_buffer.id(),
            *m_scratch,
            m_region.fold(m_op, data_type_of<T>()),
            m_memory_id
        );
        runtime.release_buffer(*m_scratch);
        m_scratch.reset();
    }

  private:
    Buffer m_buffer;
    region_type m_region;
    ReductionOp m_op;
    MemoryId m_memory_id = MemoryId::host();
    std::optional<BufferId> m_scratch;
    size_t m_index = 0;
};

}  // namespace detail

template<typename T, typename LayoutT, size_t K>
class LaunchArg<Reduce<NDArray<T, LayoutT>, K>>: public detail::LaunchArgReduce<T, LayoutT, K> {
  public:
    explicit LaunchArg(const Reduce<NDArray<T, LayoutT>, K>& arg) :
        detail::LaunchArgReduce<T, LayoutT, K>(
            arg.target.buffer(),
            arg.target.layout(),
            arg.op,
            arg.replication
        ) {}
};

template<typename T, typename LayoutT, size_t K>
class LaunchArg<Reduce<NDAccumulator<T, LayoutT>, K>>:
    public detail::LaunchArgReduce<T, LayoutT, K> {
  public:
    explicit LaunchArg(const Reduce<NDAccumulator<T, LayoutT>, K>& arg) :
        detail::LaunchArgReduce<T, LayoutT, K>(
            arg.target.buffer(),
            arg.target.layout(),
            arg.op,
            arg.replication
        ) {}
};

template<typename T, typename LayoutT>
class LaunchArg<Reduce<NDAccumulator<T, LayoutT>, 0>>: public LaunchArg<NDAccumulator<T, LayoutT>> {
  public:
    explicit LaunchArg(const Reduce<NDAccumulator<T, LayoutT>, 0>& arg) :
        LaunchArg<NDAccumulator<T, LayoutT>>(arg.target) {}
};

}  // namespace kmm
