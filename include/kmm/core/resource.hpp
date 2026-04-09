#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/core/view.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/gpu_utils.hpp"

namespace kmm {

class Resource;
class InvalidResourceException;
class ComputeTask;
struct TaskContext;

enum struct ExecutionSpace { Host, Device };

struct TaskContext {
    std::vector<BufferAccessor> accessors;
};

class ComputeTask {
  public:
    virtual ~ComputeTask() = default;
    virtual void execute(Resource& resource, TaskContext context) = 0;
};

/**
 * Exception throw if invalid resource is provided to task.
 */
class InvalidResourceException: public std::exception {
  public:
    InvalidResourceException(const std::type_info& expected, const std::type_info& gotten);
    const char* what() const noexcept override;

  private:
    std::string m_message;
};

class Resource {
  public:
    virtual ~Resource() = default;

    template<typename T>
    T* cast_if() noexcept {
        return dynamic_cast<T*>(this);
    }

    template<typename T>
    const T* cast_if() const noexcept {
        return dynamic_cast<const T*>(this);
    }

    template<typename T>
    T& cast() {
        if (auto* ptr = this->template cast_if<T>()) {
            return *ptr;
        }

        throw InvalidResourceException(typeid(T), typeid(*this));
    }

    template<typename T>
    const T& cast() const {
        if (auto* ptr = this->template cast_if<T>()) {
            return *ptr;
        }

        throw InvalidResourceException(typeid(T), typeid(*this));
    }

    template<typename T>
    bool is() const noexcept {
        return this->template cast_if<T>() != nullptr;
    }
};

class HostResource: public Resource {};

class DeviceResource: public DeviceInfo, public Resource {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceResource);

  public:
    DeviceResource(DeviceInfo info, GPUContextHandle context, g_stream_t stream);
    ~DeviceResource();

    /**
     * Returns a handle to the context associated with this device.
     */
    GPUContextHandle context_handle() const {
        return m_context;
    }

    /**
     * Returns a handle to the context associated with this device.
     */
    g_context_t context() const {
        return m_context;
    }

    /**
     * Returns a handle to the stream associated with this device.
     */
    g_stream_t stream() const {
        return m_stream;
    }

    /**
     * Shorthand for `stream()`.
     */
    operator g_stream_t() const {
        return m_stream;
    }

    /**
     * Returns a handle to the BLAS instance associated with this device.
     */
    blas_handle_t blas() const {
        return m_blas_handle;
    }

    /**
     * Block the current thread until all work submitted onto the stream of this device has
     * finished. Note that the executor thread will also synchronize the stream automatically
     * after each task, so calling thus function manually is not mandatory.
     */
    void synchronize() const;

    /**
     * Launch the given kernel onto the stream of this device. The `kernel_function` argument
     * should be a pointer to a `__global__` function.
     */
    template<typename... Args>
    void launch_impl(
        dim3 grid_dim,
        dim3 block_dim,
        unsigned int shared_mem,
        void (*const kernel_function)(Args...),
        Args... args
    ) const {
        // Get void pointer to the arguments.
        void* void_args[sizeof...(Args) + 1] = {static_cast<void*>(&args)..., nullptr};

        // Launch the kernel!
        // NOTE: This must be in the header file since `gpuLaunchKernel` seems to no find the
        // kernel function if it is called from within a C++ file.
        KMM_GPU_CHECK(gpu_launch_kernel(
            reinterpret_cast<const void*>(kernel_function),
            grid_dim,
            block_dim,
            void_args,
            shared_mem,
            m_stream
        ));
    }

    template<typename... Param, typename... Args>
    void launch(
        dim3 grid_dim,
        dim3 block_dim,
        unsigned int shared_mem,
        void (*const kernel_function)(Param...),
        Args... args
    ) const {
        launch_impl(grid_dim, block_dim, shared_mem, kernel_function, Param(args)...);
    }

    /**
     * Fill the provided buffer with the copies of the provided value. The fill is performed
     * asynchronously on the stream of this device.
     */
    template<typename T, typename I>
    void fill(T* dest, I num_elements, T value) const {
        fill_bytes(
            dest,
            checked_mul(checked_cast<size_t>(num_elements), sizeof(T)),
            &value,
            sizeof(T)
        );
    }

    /**
     * Fill the provided view with the copies of the provided value. The fill is performed
     * asynchronously on the stream of this device.
     */
    template<typename T, size_t N>
    void fill(GPUViewMut<T, N> dest, T value) const {
        KMM_ASSERT(dest.is_contiguous());
        fill_bytes(dest.data(), dest.size_in_bytes(), &value, sizeof(T));
    }

    /**
     * Copy data from the given source view to the given destination view. The copy is performed
     * asynchronously on the stream of the current device.
     */
    template<typename T, size_t N>
    void copy(GPUView<T, N> source, GPUViewMut<T, N> dest) const {
        KMM_ASSERT(source.sizes() == dest.sizes());
        KMM_ASSERT(source.is_contiguous() && dest.is_contiguous());
        copy_bytes(source.data(), dest.data(), source.size_in_bytes());
    }

    template<typename T, size_t N>
    void copy(GPUView<T, N> source, ViewMut<T, N> dest) const {
        KMM_ASSERT(source.sizes() == dest.sizes());
        KMM_ASSERT(source.is_contiguous() && dest.is_contiguous());
        copy_bytes(source.data(), dest.data(), source.size_in_bytes());
    }

    template<typename T, size_t N>
    void copy(View<T, N> source, GPUViewMut<T, N> dest) const {
        KMM_ASSERT(source.sizes() == dest.sizes());
        KMM_ASSERT(source.is_contiguous() && dest.is_contiguous());
        copy_bytes(source.data(), dest.data(), source.size_in_bytes());
    }

    /**
     * Copy data from the given source view to the given destination view. The copy is performed
     * asynchronously on the stream of the current device.
     */
    template<typename T, typename I>
    void copy(const T* source_ptr, T* dest_ptr, I num_elements) const {
        copy_bytes(
            source_ptr,
            dest_ptr,
            checked_mul(checked_cast<size_t>(num_elements), sizeof(T))
        );
    }

    /**
     * Fill `nbytes` of the buffer starting at `dest_buffer` by repeating the given pattern.
     * The argument `dest_buffer` must be allocated on the device while the `fill_pattern` must
     * be on the host.
     */
    void fill_bytes(
        void* dest_buffer,
        size_t nbytes,
        const void* fill_pattern,
        size_t fill_pattern_size
    ) const;

    /**
     * Copy `nbytes` bytes from the buffer starting at `source_buffer` to the buffer starting at
     * `dest_buffer`. Both buffers must be allocated on the current device.
     */
    void copy_bytes(const void* source_buffer, void* dest_buffer, size_t nbytes) const;

  private:
    GPUContextHandle m_context;
    g_stream_t m_stream;
    blas_handle_t m_blas_handle;
};

}  // namespace kmm
