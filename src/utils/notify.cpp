#include "kmm/utils/notify.hpp"

namespace kmm {

NotifyHandle::~NotifyHandle() = default;

void NotifyHandle::notify() const noexcept {
    if (m_impl) {
        m_impl->notify();
    }
}

void NotifyHandle::clear() noexcept {
    m_impl.reset();
}

void NotifyHandle::notify_and_clear() noexcept {
    notify();
    clear();
}

}  // namespace kmm
