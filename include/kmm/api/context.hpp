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
        KMM_ASSERT(!domain.is_empty());
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
        return Scope<Context, Args...>(
            *this,
            MemoryId::host(),
            m_transaction,
            std::forward<Args>(args)...
        );
    }

    /// Shorthand for `host_scope(args...).apply(fun)`.
    template<typename Launcher, typename... Args>
    decltype(auto) host_launch(Launcher&& launcher, Args&&... args) {
        return host_scope(std::forward<Args>(args)...).apply(std::forward<Launcher>(launcher));
    }

    virtual MemoryId affinity_memory_id() const noexcept {
        return MemoryId::host();
    }

  protected:
    template<typename, typename...>
    friend class Scope;

    template<typename...>
    friend struct ScopeImpl;

    // Rebinds `parent` to a new transaction, sharing its runtime/memory_id. Used by `Scope` to
    // give `apply`'s callback a context scoped to the requisition's own (child) transaction, and
    // by derived classes (e.g. `DeviceContext`) to do the same for their own extra state.
    Context(const Context& parent, MemoryTransaction transaction) :
        m_runtime(parent.m_runtime),
        m_transaction(std::move(transaction)) {}
    std::optional<CUDAStreamRef> submit_stream() noexcept {
        return std::nullopt;
    }

    // Hook for `Scope::apply`: makes this context's device "active" (e.g. current CUDA context)
    // on the calling thread before user code runs. The host context has nothing to activate.
    struct ActivateGuard {};
    ActivateGuard activate() const noexcept {
        return {};
    }

    // Returns a copy of this context scoped to `transaction`. `Scope` calls this (rather than
    // constructing `Ctx` itself) so a context type is free to hand back a different type here --
    // `Scope` deduces whatever this returns instead of assuming it matches `Ctx`.
    Context child(MemoryTransaction transaction) noexcept {
        return Context(*this, std::move(transaction));
    }

    Runtime m_runtime;
    MemoryTransaction m_transaction;
};

}  // namespace kmm
