#pragma once

#define KMM_INLINE   __attribute__((always_inline)) inline
#define KMM_NOINLINE __attribute__((noinline))

#define KMM_ASSUME(expr)             \
    do {                             \
        if (!(expr))                 \
            __builtin_unreachable(); \
    } while (0)
#define KMM_LIKELY(expr)   (__builtin_expect(!!(expr), true))
#define KMM_UNLIKELY(expr) (__builtin_expect(!!(expr), false))

#define KMM_CONCAT(A, B)      KMM_CONCAT_IMPL(A, B)
#define KMM_CONCAT_IMPL(A, B) A##B

#if defined(__CUDACC__)
    // CUDA
    #define KMM_HOST_DEVICE          __host__ __device__ __forceinline__
    #define KMM_DEVICE               __device__ __forceinline__
    #define KMM_HOST_DEVICE_NOINLINE __host__ __device__
    #define KMM_DEVICE_NOINLINE      __device__
    #ifdef __CUDA_ARCH__
        #define KMM_IS_DEVICE (1)

        #ifdef __CUDACC_RTC__
            #define KMM_IS_RTC (1)
        #endif
    #endif
#elif defined(__HIPCC__)
    // HIP
    // TODO: using __forceinline__ breaks compilation, it needs to be investigated
    #define KMM_HOST_DEVICE          __host__ __device__
    #define KMM_DEVICE               __device__ __forceinline__
    #define KMM_HOST_DEVICE_NOINLINE __host__ __device__
    #define KMM_DEVICE_NOINLINE      __device__
    #ifdef __HIP_DEVICE_COMPILE__
        #define KMM_IS_DEVICE (1)

        #ifdef __HIPCC_RTC__
            #define KMM_IS_RTC (1)
        #endif
    #endif
#else
    // Dummy backend
    #define KMM_HOST_DEVICE KMM_INLINE
    #define KMM_DEVICE      KMM_INLINE
    #define KMM_HOST_DEVICE_NOINLINE
    #define KMM_DEVICE_NOINLINE
    #define KMM_IS_RTC    (0)
    #define KMM_IS_DEVICE (0)
#endif

#ifdef __has_cpp_attribute
    #if __has_cpp_attribute(no_unique_address) && !(defined(__GNUC__) && (__cplusplus < 201100))
        #define KMM_ATTRIBUTE_NO_UNIQUE_ADDRESS [[no_unique_address]]
    #endif
#endif

#ifndef KMM_ATTRIBUTE_NO_UNIQUE_ADDRESS
    #define KMM_ATTRIBUTE_NO_UNIQUE_ADDRESS
#endif

#define KMM_NOT_COPYABLE(TYPE)                  \
  public:                                       \
    TYPE(const TYPE&) = delete;                 \
    TYPE& operator=(const TYPE&) = delete;      \
    TYPE(TYPE&) = delete;                       \
    TYPE& operator=(TYPE&) = delete;            \
    TYPE(TYPE&&) noexcept = default;            \
    TYPE& operator=(TYPE&&) noexcept = default; \
                                                \
  private:

#define KMM_NOT_COPYABLE_OR_MOVABLE(TYPE)      \
  public:                                      \
    TYPE(const TYPE&) = delete;                \
    TYPE& operator=(const TYPE&) = delete;     \
    TYPE(TYPE&) = delete;                      \
    TYPE& operator=(TYPE&) = delete;           \
    TYPE(TYPE&&) noexcept = delete;            \
    TYPE& operator=(TYPE&&) noexcept = delete; \
                                               \
  private:
