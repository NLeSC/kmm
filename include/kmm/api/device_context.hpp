#pragma once

#include "kmm/api/context.hpp"

namespace kmm {

class DeviceContext: public Context {
    DeviceContext(Context base, DeviceId device_id) :
        Context(std::move(base)),
        m_device_id(device_id) {}

    MemoryId preferred_memory_id() const override {
        return MemoryId::device(m_device_id);
    }

  private:
    DeviceId m_device_id;
};

}  // namespace kmm