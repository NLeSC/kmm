#pragma once

#include "kmm/core/macros.hpp"

#define KMM_PANIC(...)                                 \
    do {                                               \
        ::kmm::panic(__FILE__, __LINE__, __VA_ARGS__); \
    } while (1)

#define KMM_ASSERT(...)                                      \
    do {                                                     \
        if (KMM_UNLIKELY(!static_cast<bool>(__VA_ARGS__))) { \
            KMM_PANIC("assertion failed: " #__VA_ARGS__);    \
        }                                                    \
    } while (0)

#define KMM_DEBUG_ASSERT(...) KMM_ASSERT(__VA_ARGS__)
#define KMM_TODO()            KMM_PANIC("not implemented")

#ifndef NDEBUG
    #define KMM_UNSAFE_ASSUME(expr) KMM_ASSERT(expr)
#else
    #define KMM_UNSAFE_ASSUME(expr)      \
        do {                             \
            if (!(expr))                 \
                __builtin_unreachable(); \
        } while (0)
#endif

#if !KMM_IS_DEVICE
    #include "fmt/format.h"

    #define KMM_PANIC_FMT(...)                                                    \
        do {                                                                      \
            ::kmm::panic(__FILE__, __LINE__, ::fmt::format(__VA_ARGS__).c_str()); \
            while (1)                                                             \
                ;                                                                 \
        } while (0)
#endif

namespace kmm {

/// \addtogroup utility
/// @{

#if !KMM_IS_DEVICE
/// Logs a fatal error, prints relevant debugging info, and aborts the program.
///
/// `file` and `line` are the source location where the panic occurred, and `message` is the
/// reason for the panic.
[[noreturn]] void panic(const char* filename, int lineno, const char* message);
#else
    #if defined(__CUDACC__)
        #include <cuda_runtime.h>
    #elif defined(__HIPCC__)
        #include <hip/hip_runtime.h>
    #endif

[[noreturn]] KMM_DEVICE void panic(const char* filename, int lineno, const char* message) {
    printf(
        "[block=(%u,%u,%u) thread=(%u,%u,%u)] PANIC at %s:%d: %s\n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
        filename,
        lineno,
        message
    );

    while (true) {
        asm volatile("trap;");
    }
}
#endif

/// @}

}  // namespace kmm
