#pragma once

#include <utility>

#include "kmm/api/buffer.hpp"

namespace kmm {

class ArrayBase {
  public:
    ArrayBase() = default;
    ArrayBase(const ArrayBase&) = default;

    const Buffer& buffer() const {
        return m_buffer;
    }

    Runtime runtime() const noexcept {
        return m_buffer.runtime();
    }

    explicit operator bool() const {
        return bool(m_buffer);
    }

  protected:
    explicit ArrayBase(Buffer buffer) noexcept : m_buffer(std::move(buffer)) {}

  private:
    Buffer m_buffer;
};

}  // namespace kmm
