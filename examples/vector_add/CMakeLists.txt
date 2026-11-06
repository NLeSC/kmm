add_executable(vector_add ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)

if(KMM_USE_CUDA)
    set_target_properties(
            vector_add
            PROPERTIES
            CUDA_ARCHITECTURES "80"
    )
    # KMM_LAMBDA expands to a __device__-annotated lambda, which nvcc only accepts with this flag.
    target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

if(KMM_USE_HIP)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.cu PROPERTIES LANGUAGE HIP)
endif()

target_compile_features(vector_add PRIVATE cxx_std_17)
target_link_libraries(vector_add PRIVATE kmm)
