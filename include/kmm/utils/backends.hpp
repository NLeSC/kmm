#pragma once

#ifdef KMM_USE_CUDA
    #include "kmm/utils/backends/cuda.hpp"
#elif KMM_USE_HIP
    #include "kmm/utils/backends/hip.hpp"
#else
    #include "kmm/utils/backends/cpu.hpp"
#endif
