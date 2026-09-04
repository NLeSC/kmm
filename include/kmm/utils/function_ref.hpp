#pragma once

#include <type_traits>
#include <utility>

namespace kmm {

/// \addtogroup utility
/// @{

template<typename Fn>
class function_ref;

/**
 * A lightweight, non-owning reference to a callable, similar to `std::function_ref`.
 */
template<typename R, typename... Args>
class function_ref<R(Args...)> {
  public:
    function_ref() noexcept = default;
    function_ref(decltype(nullptr)) noexcept {}

    template<
        typename Fn,
        typename = std::enable_if_t<
            !std::is_same_v<std::decay_t<Fn>, function_ref>
            && std::is_invocable_r_v<R, Fn&, Args...>>>
    function_ref(Fn&& callable) noexcept :
        m_ptr(reinterpret_cast<void*>(std::addressof(callable))),
        m_invoke(&invoke<std::remove_reference_t<Fn>>) {}

    R operator()(Args... args) const {
        return m_invoke(m_ptr, std::forward<Args>(args)...);
    }

    explicit operator bool() const noexcept {
        return m_ptr != nullptr;
    }

  private:
    template<typename Fn>
    static R invoke(void* ptr, Args... args) {
        return (*reinterpret_cast<Fn*>(ptr))(std::forward<Args>(args)...);
    }

    void* m_ptr = nullptr;
    R (*m_invoke)(void*, Args...) = nullptr;
};

/// @}

}  // namespace kmm
