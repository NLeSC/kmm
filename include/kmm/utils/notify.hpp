#pragma once

#include <memory>

#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// \addtogroup utility
/// @{

/**
 * Simple interface having a `notify` method to be called when an event happens.
 */
class Notify: public reference_count<Notify> {
  public:
    virtual ~Notify() noexcept = default;
    virtual void notify() const noexcept = 0;
};

/**
 * Wrapper around a `std::shared_ptr<Notify>`.
 */
class NotifyHandle {
  public:
    NotifyHandle() = default;
    ~NotifyHandle();

    /** Constructs an empty (null) handle. */
    NotifyHandle(decltype(nullptr)) {};

    /** Constructs a handle from a shared pointer to a `Notify` instance. */
    NotifyHandle(refcnt_ptr<const Notify> m) : m_impl(std::move(m)) {}

    /** Constructs a handle by taking ownership of a `Notify` via unique pointer. */
    NotifyHandle(std::unique_ptr<const Notify> m) : m_impl(std::move(m)) {}

    template<typename T>
    NotifyHandle(refcnt_ptr<T> m) : m_impl(std::move(m)) {}

    template<typename T>
    NotifyHandle(std::unique_ptr<T> m) : m_impl(std::move(m)) {}

    /// Constructs a handle from any callable (e.g. lambda, functor, function pointer).
    /// The callable is wrapped in an internal `Notify` implementation and stored via shared pointer.
    template<typename F, typename = std::enable_if_t<std::is_invocable_v<std::decay_t<F>>>>
    NotifyHandle(F&& callback) :
        m_impl(make_refcnt<Impl<std::decay_t<F>>>(std::forward<F>(callback))) {}

    /// Calls `notify` on the inner notifier. Does nothing if the handle is empty.
    void notify() const noexcept;

    /// Resets the handle, releasing the reference to the inner notifier.
    void clear() noexcept;

    /// Notifies and then clears the handle. Equivalent to `notify(); clear();`.
    void notify_and_clear() noexcept;

    /// Returns `true` if the handle holds a notifier, `false` if it is empty.
    explicit operator bool() const noexcept {
        return m_impl != nullptr;
    }

  private:
    // Internal `Notify` implementation that wraps a callable `F`.
    template<typename F>
    class Impl: public Notify {
      public:
        explicit Impl(F fun) : m_callback(std::move(fun)) {}

        void notify() const noexcept override {
            m_callback();
        }

      private:
        F m_callback;
    };

    refcnt_ptr<const Notify> m_impl;
};

/// @}

}  // namespace kmm