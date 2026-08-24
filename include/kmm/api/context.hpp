#pragma once

#include <optional>
#include <utility>

#include "kmm/api/array.hpp"
#include "kmm/api/scope.hpp"
#include "kmm/runtime/identifiers.hpp"
#include "kmm/runtime/memory_manager.hpp"
#include "kmm/runtime/requisition.hpp"
#include "kmm/runtime/runtime.hpp"

namespace kmm {

class DeviceContext;

class Context {
  public:
    Context(Runtime runtime, MemoryTransaction transaction = {}) :
        m_runtime(std::move(runtime)),
        m_transaction(std::move(transaction)) {}

    template<typename T, typename DomainT, typename PolicyT = RowMajor>
    DomainArray<T, DomainT, PolicyT> array(
        DomainT domain,
        PolicyT policy = {},
        std::optional<T> fill_value = {}
    ) {
        return DomainArray<T, DomainT, PolicyT>(
            runtime(),
            domain,
            policy,
            fill_value,
            affinity_memory_id()
        );
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> empty(ExtentT... extents) {
        return array<T>(shape(extents...), PolicyT {});
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> fill(T value, ExtentT... extents) {
        return array(shape(extents...), PolicyT {}, value);
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> ones(ExtentT... extents) {
        return array(shape(extents...), PolicyT {}, static_cast<T>(1));
    }

    template<typename T, typename PolicyT = RowMajor, typename... ExtentT>
    Array<T, sizeof...(ExtentT)> zeros(ExtentT... extents) {
        return array(shape(extents...), PolicyT {}, T {});
    }

    template<typename T, typename DomainT, typename PolicyT>
    DomainArray<T, DomainT, PolicyT> empty_like(const DomainArray<T, DomainT, PolicyT>& that) {
        return array<T>(that.domain(), PolicyT {});
    }

    template<typename T, typename DomainT, typename PolicyT>
    DomainArray<T, DomainT, PolicyT> ones_like(const DomainArray<T, DomainT, PolicyT>& that) {
        return array<T>(that.domain(), PolicyT {}, static_cast<T>(1));
    }

    template<typename T, typename DomainT, typename PolicyT>
    DomainArray<T, DomainT, PolicyT> zeros_like(const DomainArray<T, DomainT, PolicyT>& that) {
        return array<T>(that.domain(), PolicyT {}, T {});
    }

    template<typename T, typename DomainT, typename PolicyT>
    DomainArray<T, DomainT, PolicyT> copy(const DomainArray<T, DomainT, PolicyT>& that) {
        auto result = empty_like(that);
        copy(result, that);
        return result;
    }

    template<
        typename T,
        typename DstDomainT,
        typename DstPolicyT,
        typename SrcDomainT,
        typename SrcPolicyT>
    DeviceEvent copy(
        DomainArray<T, DstDomainT, DstPolicyT>& dst,
        const DomainArray<T, SrcDomainT, SrcPolicyT>& src
    ) {
        return copy(dst, src, affinity_memory_id());
    }

    template<
        typename T,
        typename DstDomainT,
        typename DstPolicyT,
        typename SrcDomainT,
        typename SrcPolicyT>
    DeviceEvent copy(
        DomainArray<T, DstDomainT, DstPolicyT>& dst,
        const DomainArray<T, SrcDomainT, SrcPolicyT>& src,
        MemoryId dst_memory_id
    ) {
        return m_runtime.submit_copy(
            dst.buffer().id(),
            src.buffer().id(),
            make_copy_description(dst.layout(), src.layout(), sizeof(T)),
            dst_memory_id,
            transaction()
        );
    }

    template<typename T>
    Array<T> from_host(const T* data, size_t length) {
        auto result = empty<T>(length);
        result.buffer().copy_from(data, length * sizeof(T));
        return result;
    }

    template<typename T>
    Array<T> from_host(const std::vector<T>& data) {
        return from_host(data.data(), data.size());
    }

    void prefetch(const Buffer& buffer, bool invalidate_others = false) {
        buffer.prefetch(affinity_memory_id(), invalidate_others);
    }

    template<typename T, typename DomainT, typename PolicyT>
    void prefetch(const DomainArray<T, DomainT, PolicyT>& src, bool invalidate_others = false) {
        prefetch(src.buffer(), invalidate_others);
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

    DeviceContext gpu(DeviceId device_id = DeviceId(0));

    template<typename... Args>
    Scope<Context, Args...> host_scope(Args&&... args) {
        auto scope = std::make_unique<ScopeImpl<Args...>>(
            m_runtime,
            MemoryId::host(),
            std::nullopt,
            m_transaction,
            std::forward<Args>(args)...
        );
        auto context = std::make_unique<Context>(m_runtime, scope->transaction());
        return Scope<Context, Args...>(std::move(scope), std::move(context));
    }

    /// Shorthand for `host_scope(args...).apply(fun)`.
    template<typename Launcher, typename... Args>
    decltype(auto) host_launch(Launcher&& launcher, Args&&... args) {
        return host_scope(std::forward<Args>(args)...).apply(std::forward<Launcher>(launcher));
    }

    virtual MemoryId affinity_memory_id() const noexcept {
        return MemoryId::host();
    }

    Runtime m_runtime;
    MemoryTransaction m_transaction;
};

template<>
struct scope_invoke<Context> {
    template<typename ScopeT, typename F>
    void operator()(Context& context, ScopeT& scope, DeviceEventSet& deps, F&& fun) const {
        context.runtime().synchronize(scope.dependencies());
        scope.apply(std::forward<F>(fun), context);
    }
};

}  // namespace kmm
