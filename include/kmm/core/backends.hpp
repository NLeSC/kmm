#pragma once

#ifdef KMM_USE_CUDA
    #include "kmm/backends/cuda.hpp"
#elif KMM_USE_HIP
    #include "kmm/backends/hip.hpp"
#else
    #include "kmm/backends/cpu.hpp"
#endif
