#pragma once

#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

#include "fmt/ostream.h"

#include "kmm/core/panic.hpp"

namespace kmm {

/// \addtogroup utility
/// @{

[[noreturn]] __attribute__((noinline)) void throw_small_vector_out_of_capacity();

template<typename T, size_t InlineSize>
struct small_vector {
    static_assert(
        std::is_nothrow_default_constructible_v<T> && std::is_nothrow_move_constructible_v<T>
            && std::is_nothrow_move_assignable_v<T>,
        "small_vector requires T to be nothrow default constructible, nothrow move constructible, "
        "and nothrow move assignable"
    );

    using capacity_type = uint32_t;

    small_vector() = default;

    small_vector(const small_vector& that) {
        insert_all(that.begin(), that.end());
    }

    template<typename U, size_t N>
    small_vector(const small_vector<U, N>& that) {
        insert_all(that);
    }

    small_vector(small_vector&& that) noexcept {
        *this = std::move(that);
    }

    small_vector(std::initializer_list<T> items) {
        insert_all(items.begin(), items.end());
    }

    small_vector& operator=(const small_vector& that) {
        if (this != &that) {
            clear();
            insert_all(that);
        }

        return *this;
    }

    template<typename U, size_t K>
    small_vector& operator=(const small_vector<U, K>& that) {
        clear();
        insert_all(that);
        return *this;
    }

    small_vector& operator=(small_vector&& that) noexcept {
        swap(that);
        return *this;
    }

    void swap(small_vector& that) noexcept {
        std::swap(this->m_inline_data, that.m_inline_data);
        std::swap(this->m_size, that.m_size);
        std::swap(this->m_capacity, that.m_capacity);
        std::swap(this->m_data, that.m_data);

        if (!this->is_heap_allocated()) {
            this->m_data = this->m_inline_data;
        }

        if (!that.is_heap_allocated()) {
            that.m_data = that.m_inline_data;
        }
    }

    friend void swap(small_vector& a, small_vector& b) noexcept {
        a.swap(b);
    }

    ~small_vector() noexcept {
        if (is_heap_allocated()) {
            delete[] m_data;
        }
    }

    size_t capacity() const noexcept {
        return m_capacity;
    }

    size_t size() const noexcept {
        return m_size;
    }

    bool is_empty() const noexcept {
        return m_size == 0;
    }

    bool is_heap_allocated() const noexcept {
        return m_capacity > InlineSize;
    }

    T* data() noexcept {
        return m_data;
    }

    const T* data() const noexcept {
        return m_data;
    }

    bool try_grow_capacity(size_t k = 1) noexcept {
        capacity_type new_capacity = m_capacity;

        if (new_capacity < 16) {
            new_capacity = 16;
        }

        while (new_capacity - m_size < k) {
            if (new_capacity > std::numeric_limits<capacity_type>::max() - new_capacity) {
                // Failed to grow capacity
                return false;
            }

            new_capacity += new_capacity;
        }

        auto new_data = std::unique_ptr<T[]>(new (std::nothrow_t {}) T[new_capacity] {});

        if (new_data == nullptr) {
            return false;
        }

        for (size_t i = 0; i < m_size; i++) {
            new_data[i] = std::move(m_data[i]);
        }

        if (is_heap_allocated()) {
            delete[] m_data;
        } else {
            // The inline slots are not deallocated like a heap array would be, so drop
            // their moved-from contents now instead of letting them linger unused.
            for (size_t i = 0; i < m_size; i++) {
                m_inline_data[i] = T {};
            }
        }

        m_capacity = new_capacity;
        m_data = new_data.release();
        return true;
    }

    bool try_push_back(T item) noexcept {
        if (m_capacity <= m_size && !try_grow_capacity(1)) {
            return false;
        }

        m_data[m_size] = std::move(item);
        m_size++;
        return true;
    }

    void push_back(T item) {
        if (!try_push_back(std::move(item))) {
            throw_small_vector_out_of_capacity();
        }
    }

    template<typename It>
    void insert_all(It begin, It end) {
        size_t n = static_cast<size_t>(end - begin);

        if (KMM_UNLIKELY(m_capacity - m_size < n)) {
            // FIX: what if begin,end points to this small_vector, this will invalidate pointers
            if (!try_grow_capacity(n)) {
                throw_small_vector_out_of_capacity();
            }
        }

        for (size_t i = 0; i < n; i++) {
            m_data[m_size + i] = *(begin + i);
        }

        // This is safe since `n <= m_capacity - m_size`
        m_size += static_cast<capacity_type>(n);
    }

    void insert_all(small_vector&& that) {
        if (is_empty()) {
            *this = std::move(that);
            return;
        }

        insert_all(that.begin(), that.end());
    }

    template<typename U, size_t K>
    void insert_all(const small_vector<U, K>& that) {
        insert_all(that.begin(), that.end());
    }

    void resize(size_t n) {
        if (n < m_size) {
            truncate(n);
            return;
        }

        if (m_capacity < n) {
            if (!try_grow_capacity(n - m_size)) {
                throw_small_vector_out_of_capacity();
            }
        }

        // Safe since `n <= m_capacity`
        m_size = static_cast<capacity_type>(n);
    }

    void truncate(size_t new_size) noexcept {
        if (new_size >= m_size) {
            return;
        }

        // Drop the trailing elements so any resource they hold is released now,
        // rather than lingering until the slot is reused or the vector is destroyed.
        for (size_t i = new_size; i < m_size; i++) {
            m_data[i] = T {};
        }

        m_size = static_cast<capacity_type>(new_size);
    }

    void clear() noexcept {
        truncate(0);
    }

    T& operator[](size_t i) noexcept {
        return *(data() + i);
    }

    T* begin() noexcept {
        return data();
    }

    T* end() noexcept {
        return data() + size();
    }

    const T& operator[](size_t i) const noexcept {
        return *(data() + i);
    }

    const T* begin() const noexcept {
        return data();
    }

    const T* end() const noexcept {
        return data() + size();
    }

  private:
    T* m_data = m_inline_data;
    capacity_type m_size = 0;
    capacity_type m_capacity = InlineSize;
    T m_inline_data[InlineSize] {};
};

using byte_buffer = small_vector<uint8_t, sizeof(uint64_t)>;

template<typename T, size_t N>
std::ostream& operator<<(std::ostream& stream, const small_vector<T, N>& p) {
    stream << "{";
    for (size_t i = 0; i < p.size(); i++) {
        if (i != 0) {
            stream << ", ";
        }

        stream << p[i];
    }

    return stream << "}";
}

/// @}

}  // namespace kmm

template<typename T, size_t N>
struct fmt::formatter<kmm::small_vector<T, N>>: fmt::ostream_formatter {};