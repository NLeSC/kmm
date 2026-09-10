#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace kmm {

template<typename T>
struct reference_count {
  protected:
    constexpr reference_count() noexcept = default;

  private:
    template<typename U>
    friend struct refcnt_traits_impl;

    mutable std::atomic<size_t> m_count {1};
};

template<typename T>
struct refcnt_traits_impl {
    template<typename U, typename = std::enable_if_t<std::is_base_of_v<U, T>>>
    static void increment(const reference_count<U>* that) noexcept {
        that->m_count.fetch_add(1, std::memory_order_relaxed);
    }

    template<typename U, typename = std::enable_if_t<std::is_base_of_v<U, T>>>
    static void decrement(const reference_count<U>* that) noexcept {
        if (that->m_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete static_cast<const T*>(that);
        }
    }

    template<typename U, typename = std::enable_if_t<std::is_base_of_v<U, T>>>
    static size_t load_count(const reference_count<U>* that) noexcept {
        return that->m_count.load(std::memory_order_acquire);
    }
};

template<typename T>
struct refcnt_traits final: refcnt_traits_impl<T> {};

/// Declares (without defining) an explicit specialization of `refcnt_traits<T>`, so that
/// `refcnt_ptr<T>` can be used in a header where `T` is still an incomplete type. Pair with
/// `KMM_REFCNT_TRAITS_IMPL(T)` in a source file where `T` is complete.
#define KMM_REFCNT_TRAITS_FWD(T)                          \
    template<>                                            \
    struct refcnt_traits<T> final {                       \
        static void increment(const T* that) noexcept;    \
        static void decrement(const T* that) noexcept;    \
        static size_t load_count(const T* that) noexcept; \
    };

#define KMM_REFCNT_TRAITS_IMPL(T)                                      \
    void ::kmm::refcnt_traits<T>::increment(const T* that) noexcept {  \
        ::kmm::refcnt_traits_impl<T>::increment(that);                 \
    }                                                                  \
    void ::kmm::refcnt_traits<T>::decrement(const T* that) noexcept {  \
        ::kmm::refcnt_traits_impl<T>::decrement(that);                 \
    }                                                                  \
    size_t kmm::refcnt_traits<T>::load_count(const T* that) noexcept { \
        return ::kmm::refcnt_traits_impl<T>::load_count(that);         \
    }

template<typename T, typename Traits = refcnt_traits<T>>
class refcnt_ptr {
    using pointer = std::add_pointer_t<T>;
    using element_type = T;
    using traits_type = Traits;

  public:
    refcnt_ptr() noexcept = default;

    explicit refcnt_ptr(pointer p, bool increment_ref) noexcept : m_ptr(p) {
        if (increment_ref && *this) {
            traits_type::increment(this->get());
        }
    }

    refcnt_ptr(std::unique_ptr<T>&& other) noexcept : refcnt_ptr(other.release(), false) {}

    refcnt_ptr(const refcnt_ptr& other) noexcept : m_ptr(other.m_ptr) {
        if (*this) {
            traits_type::increment(this->get());
        }
    }

    refcnt_ptr(refcnt_ptr&& other) noexcept : m_ptr(other.release()) {}

    template<typename U, typename UTraits, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
    refcnt_ptr(const refcnt_ptr<U, UTraits>& other) noexcept : m_ptr(other.get()) {
        if (other) {
            UTraits::increment(other.get());
        }
    }

    template<typename U, typename UTraits, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
    refcnt_ptr(refcnt_ptr<U, UTraits>&& other) noexcept : m_ptr(other.release()) {}

    refcnt_ptr(std::nullptr_t) noexcept : refcnt_ptr() {}

    ~refcnt_ptr() {
        if (*this) {
            traits_type::decrement(this->get());
        }
    }

    refcnt_ptr& operator=(const refcnt_ptr& other) noexcept {
        if (&other != this) {
            if (other) {
                traits_type::increment(other.get());
            }
            if (*this) {
                traits_type::decrement(this->get());
            }
            this->m_ptr = other.get();
        }
        return *this;
    }

    refcnt_ptr& operator=(refcnt_ptr&& other) noexcept {
        if (&other != this) {
            this->swap(other);
        }
        return *this;
    }

    refcnt_ptr& operator=(std::nullptr_t) noexcept {
        this->reset();
        return *this;
    }

    element_type& operator*() const noexcept {
        return *(this->get());
    }

    pointer operator->() const noexcept {
        return this->get();
    }

    pointer get() const noexcept {
        return this->m_ptr;
    }

    explicit operator bool() const noexcept {
        return this->get() != nullptr;
    }

    /// Returns true if this is the only `refcnt_ptr` referring to the pointee. Note that this
    /// is a snapshot: without external synchronization, another thread may concurrently copy or
    /// drop a `refcnt_ptr` to the same object, invalidating the result immediately after it is read.
    bool unique() const noexcept {
        return *this && traits_type::load_count(this->get()) == 1;
    }

    pointer release() noexcept {
        auto* ptr = this->get();
        this->m_ptr = pointer {};
        return ptr;
    }

    void reset(pointer p = pointer {}) noexcept {
        *this = refcnt_ptr(p, false);
    }

    void swap(refcnt_ptr& other) noexcept {
        using std::swap;
        swap(m_ptr, other.m_ptr);
    }

  private:
    pointer m_ptr = nullptr;
};

template<typename T, typename Traits, typename U, typename UTraits>
bool operator==(const refcnt_ptr<T, Traits>& x, const refcnt_ptr<U, UTraits>& y) noexcept {
    return x.get() == y.get();
}

template<typename T, typename Traits, typename U, typename UTraits>
bool operator!=(const refcnt_ptr<T, Traits>& x, const refcnt_ptr<U, UTraits>& y) noexcept {
    return !(x == y);
}

template<typename T, typename Traits>
bool operator==(const refcnt_ptr<T, Traits>& x, std::nullptr_t) noexcept {
    return x.get() == nullptr;
}

template<typename T, typename Traits>
bool operator!=(const refcnt_ptr<T, Traits>& x, std::nullptr_t) noexcept {
    return x.get() != nullptr;
}

template<typename T, typename Traits>
bool operator==(std::nullptr_t, const refcnt_ptr<T, Traits>& x) noexcept {
    return x.get() == nullptr;
}

template<typename T, typename Traits>
bool operator!=(std::nullptr_t, const refcnt_ptr<T, Traits>& x) noexcept {
    return x.get() != nullptr;
}

template<typename T, typename... Args>
refcnt_ptr<T> make_refcnt(Args&&... args) {
    return refcnt_ptr<T>(new T(std::forward<Args>(args)...), false);
}

}  // namespace kmm
