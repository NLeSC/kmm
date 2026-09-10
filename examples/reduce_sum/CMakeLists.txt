add_executable(reduce_sum ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)

if(KMM_USE_CUDA)
    set_target_properties(
            reduce_sum
            PROPERTIES
            CUDA_ARCHITECTURES "80"
    )
endif()

if(KMM_USE_HIP)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.cu PROPERTIES LANGUAGE HIP)
endif()

target_compile_features(reduce_sum PRIVATE cxx_std_17)
target_link_libraries(reduce_sum PRIVATE kmm)
