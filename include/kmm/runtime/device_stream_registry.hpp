#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <optional>

#include "kmm/runtime/device_stream.hpp"
#include "kmm/utils/function_ref.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/refcnt_ptr.hpp"

namespace kmm {

/// \addtogroup runtime
/// @{

class DeviceStreamRegistry {
  public:
    DeviceStreamRegistry();
    ~DeviceStreamRegistry();

    DeviceStream lookup(CUstream stream);
    DeviceStream create(CUcontext context, bool high_priority = false);
    void synchronize();
    void make_progress();

  public:
    struct Impl;

  private:
    std::unique_ptr<Impl> m_impl;
};

/// @}

KMM_REFCNT_TRAITS_FWD(DeviceStreamRegistry::Impl)

}  // namespace kmm