#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <utility>

#include "kmm/api/accumulator.hpp"
#include "kmm/api/array.hpp"
#include "kmm/api/reduce.hpp"
#include "kmm/api/resource_guard.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memops/reduction.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/resource.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

class Host;
class Device;

class Context {
  public:
    Context(Runtime runtime, MemoryTransaction transaction = {}) :
        m_runtime(std::move(runtime)),
        m_transaction(std::move(transaction)) {}

    template<typename T, typename DomainT, typename PolicyT = RowMajor>
    NDArray<T, Layout<DomainT, PolicyT>> array(
        DomainT domain,
        PolicyT policy = {},
        std::optional<T> fill_value = {}
    ) {
        return NDArray<T, Layout<DomainT, PolicyT>>(
            runtime(),
            domain,
            policy,
            fill_value,
            affinity_memory_id()
        );
    }

    template<typename T, typename DomainT, typename PolicyT = RowMajor>
    NDArray<T, Layout<DomainT, PolicyT>> adopt(
        T* data,
        DomainT domain,
        std::optional<MemoryId> memory_id = {},
        PolicyT policy = {}
    ) {
        return NDArray<T, Layout<DomainT, PolicyT>>(
            runtime(),
            make_layout(domain, policy),
            data,
            memory_id.value_or(affinity_memory_id())
        );
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> empty(ExtentT... extents) {
        return array<T>(shape(extents...), PolicyT {});
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> fill(T value, ExtentT... extents) {
        return array<T>(shape(extents...), PolicyT {}, value);
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> ones(ExtentT... extents) {
        return array<T>(shape(extents...), PolicyT {}, static_cast<T>(1));
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> zeros(ExtentT... extents) {
        return array<T>(shape(extents...), PolicyT {}, T {});
    }

    template<typename T, typename LayoutT>
    NDArray<T, LayoutT> empty_like(const NDArray<T, LayoutT>& that) {
        return array<T>(that.domain(), typename LayoutT::policy_type {});
    }

    template<typename T, typename LayoutT>
    NDArray<T, LayoutT> ones_like(const NDArray<T, LayoutT>& that) {
        return array<T>(that.domain(), typename LayoutT::policy_type {}, static_cast<T>(1));
    }

    template<typename T, typename LayoutT>
    NDArray<T, LayoutT> zeros_like(const NDArray<T, LayoutT>& that) {
        return array<T>(that.domain(), typename LayoutT::policy_type {}, T {});
    }

    template<typename T>
    Array<T> from_vector(const T* data, size_t nelem) {
        auto result = empty<T>(nelem);
        result.buffer().copy_from(data, nelem * sizeof(T), 0, affinity_memory_id());
        return result;
    }

    template<typename T>
    Array<T> from_vector(const std::vector<T>& input) {
        return from_vector(input.data(), input.size());
    }

    template<typename T>
    std::vector<T> to_vector(Array<T> input) {
        size_t offset = checked_cast<size_t>(input.layout().base_offset());
        size_t nelem = checked_cast<size_t>(input.layout().size());

        std::vector<T> result;
        result.resize(nelem);
        input.buffer()
            .copy_to(result.data(), nelem * sizeof(T), offset * sizeof(T), affinity_memory_id());
        return result;
    }

    template<typename T>
    Array<T> from_scalar(const T& value) {
        return from_vector(&value, 1);
    }

    template<typename T, typename LayoutT>
    T to_scalar(const NDArray<T, LayoutT>& input) {
        KMM_ASSERT(input.layout().size() == 1);
        size_t offset = checked_cast<size_t>(input.layout().base_offset());

        T result;
        input.buffer().copy_to(&result, sizeof(T), offset * sizeof(T), affinity_memory_id());
        return result;
    }

    template<typename T, typename LayoutT>
    NDArray<T, LayoutT> copy(const NDArray<T, LayoutT>& that) {
        auto result = empty_like(that);
        copy(result, that);
        return result;
    }

    template<typename T, typename DstLayoutT, typename SrcLayoutT>
    DeviceEvent copy(NDArray<T, DstLayoutT>& dst, const NDArray<T, SrcLayoutT>& src) {
        return copy(dst, src, affinity_memory_id());
    }

    template<typename T, typename DstLayoutT, typename SrcLayoutT>
    DeviceEvent copy(
        NDArray<T, DstLayoutT>& dst,
        const NDArray<T, SrcLayoutT>& src,
        MemoryId dst_memory_id
    ) {
        return m_runtime.submit_copy(
            dst.buffer().id(),
            src.buffer().id(),
            make_copy_description(dst.layout(), src.layout(), sizeof(T)),
            dst_memory_id,
            std::nullopt,
            transaction()
        );
    }

    void prefetch(const Buffer& buffer, bool invalidate_others = false) {
        buffer.prefetch(affinity_memory_id(), invalidate_others);
    }

    template<typename T, typename LayoutT>
    void prefetch(const NDArray<T, LayoutT>& src, bool invalidate_others = false) {
        prefetch(src.buffer(), invalidate_others);
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    NDAccumulator<T, Layout<Shape<sizeof...(ExtentT)>, PolicyT>> accumulator(
        ReductionOp op,
        ExtentT... extents
    ) {
        return NDAccumulator<T, Layout<Shape<sizeof...(ExtentT)>, PolicyT>>(
            empty<T, PolicyT>(extents...),
            op
        );
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    NDAccumulator<T, Layout<Shape<sizeof...(ExtentT)>, PolicyT>> sum_accumulator(ExtentT... extents
    ) {
        return accumulator<T, PolicyT>(ReductionOp::Sum, extents...);
    }

    const Runtime& runtime() const noexcept {
        return m_runtime;
    }

    Runtime& runtime() noexcept {
        return m_runtime;
    }

    const MemoryTransaction& transaction() const noexcept {
        return m_transaction;
    }

    const SystemInfo& system_info() const noexcept {
        return m_runtime.system_info();
    }

    void synchronize() {
        m_runtime.synchronize();
    }

    void synchronize(const DeviceEvent& event) {
        m_runtime.synchronize(event);
    }

    void synchronize(const DeviceEventSet& events) {
        m_runtime.synchronize(events);
    }

    /// Returns a `Host` context sharing this context's runtime and transaction.
    Host host();

    /// Returns a `Device` context targeting `device_id`, sharing this context's runtime and
    /// transaction.
    Device gpu(DeviceId device_id = DeviceId(0));

    virtual MemoryId affinity_memory_id() const noexcept {
        return MemoryId::host();
    }

    virtual std::optional<GPUStreamRef> affinity_stream() const noexcept {
        return std::nullopt;
    }

  protected:
    Runtime m_runtime;
    MemoryTransaction m_transaction;
};

}  // namespace kmm
