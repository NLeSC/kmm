#pragma once

#include <type_traits>
#include <utility>

#include "kmm/core/macros.hpp"

namespace kmm {

/// \addtogroup utility
/// @{

/**
 * Invokes a callable when it goes out of scope, unless it has been released.
 *
 * Modeled after `std::experimental::scope_exit` from the Library Fundamentals TS.
 * Useful for running cleanup code on every exit path from a scope, including
 * early returns and exceptions.
 *
 * @code
 * auto guard = scope_exit([&] { release(handle); });
 * // ...
 * guard.release();  // optional: skip the cleanup
 * @endcode
 */
template<typename Fn>
class scope_exit {
    KMM_NOT_COPYABLE_OR_MOVABLE(scope_exit)

  public:
    explicit scope_exit(Fn fn) noexcept(std::is_nothrow_move_constructible_v<Fn>) :
        m_fn(std::move(fn)) {}

    ~scope_exit() {
        if (m_active) {
            m_fn();
        }
    }

    /**
     * Cancel the pending invocation, so the callable is not run on destruction.
     */
    void release() noexcept {
        m_active = false;
    }

  private:
    Fn m_fn;
    bool m_active = true;
};

template<typename Fn>
scope_exit(Fn) -> scope_exit<Fn>;

/// @}

}  // namespace kmm
